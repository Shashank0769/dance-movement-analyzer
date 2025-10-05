# app/pose_utils.py
import math
import numpy as np

# Map MediaPipe indices to friendly names where needed
# (Using mp_pose landmark index convention is optional; here we accept dicts keyed by name)
LANDMARK_NAMES = [
    "nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right","left_shoulder","right_shoulder","left_elbow",
    "right_elbow","left_wrist","right_wrist","left_pinky","right_pinky","left_index","right_index",
    "left_thumb","right_thumb","left_hip","right_hip","left_knee","right_knee","left_ankle",
    "right_ankle","left_heel","right_heel","left_foot_index","right_foot_index"
]

def extract_landmarks(pose_landmarks, frame_shape):
    """
    Convert MediaPipe landmarks to a dict: name -> (x_px, y_px, z, visibility)
    Returns None for missing landmarks (if pose_landmarks is None)
    """
    if pose_landmarks is None:
        return None
    h, w = frame_shape[0], frame_shape[1]
    lm = {}
    for idx, l in enumerate(pose_landmarks.landmark):
        name = LANDMARK_NAMES[idx] if idx < len(LANDMARK_NAMES) else f"lm_{idx}"
        lm[name] = (l.x * w, l.y * h, l.z, getattr(l, "visibility", 1.0))
    return lm

def angle_between(a, b, c):
    """Return angle (degrees) at point b formed by points a-b-c"""
    if a is None or b is None or c is None:
        return None
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    cx, cy = c[0], c[1]
    ab = (ax - bx, ay - by)
    cb = (cx - bx, cy - by)
    dot = ab[0]*cb[0] + ab[1]*cb[1]
    mag = math.hypot(*ab) * math.hypot(*cb)
    if mag == 0:
        return None
    cosang = max(min(dot/mag, 1.0), -1.0)
    return math.degrees(math.acos(cosang))

def detect_poses_from_landmarks(lm):
    """
    Rule-based detection:
    - t_pose: both wrists approx horizontal with shoulders and wrists away from body
    - hands_up: both wrists y < shoulders y (higher in image coordinates smaller y)
    - squat: knee angle small (hip-knee-ankle angle ~ < 120)
    - step_left / step_right: hip x displacement significant (requires sequential logic; we approximate)
    Returns list of labels for this frame.
    """
    if lm is None:
        return []

    # helper to safe-get
    def g(name):
        return lm.get(name)

    labels = []

    # Hands-up
    left_wrist = g("left_wrist")
    right_wrist = g("right_wrist")
    left_shoulder = g("left_shoulder")
    right_shoulder = g("right_shoulder")
    if left_wrist and right_wrist and left_shoulder and right_shoulder:
        # in image coords, y increases downward; hands-up -> wrist y smaller than shoulder y
        if left_wrist[1] < left_shoulder[1] - 15 and right_wrist[1] < right_shoulder[1] - 15:
            labels.append("hands_up")

    # T-pose: wrists roughly horizontally aligned with shoulders and fairly far out
    if left_wrist and right_wrist and left_shoulder and right_shoulder:
        left_dx = abs(left_wrist[0] - left_shoulder[0])
        right_dx = abs(right_wrist[0] - right_shoulder[0])
        left_dy = abs(left_wrist[1] - left_shoulder[1])
        right_dy = abs(right_wrist[1] - right_shoulder[1])
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0]) + 1e-6
        if left_dx > 0.7 * shoulder_width and right_dx > 0.7 * shoulder_width and left_dy < 0.25 * shoulder_width and right_dy < 0.25 * shoulder_width:
            labels.append("t_pose")

    # Squat (approx via knee angle)
    left_hip, right_hip = g("left_hip"), g("right_hip")
    left_knee, right_knee = g("left_knee"), g("right_knee")
    left_ankle, right_ankle = g("left_ankle"), g("right_ankle")
    angles = []
    left_angle = angle_between(left_hip, left_knee, left_ankle)
    right_angle = angle_between(right_hip, right_knee, right_ankle)
    if left_angle:
        angles.append(left_angle)
    if right_angle:
        angles.append(right_angle)
    if angles:
        avg_knee_angle = sum(angles) / len(angles)
        if avg_knee_angle < 120:  # approximate threshold
            labels.append("squat")

    # Step detection naive: if one ankle is far forward/back relative to hips we can label step-left/right
    if left_ankle and right_ankle and left_hip and right_hip:
        hip_center_x = (left_hip[0] + right_hip[0]) / 2
        left_dx = left_ankle[0] - hip_center_x
        right_dx = right_ankle[0] - hip_center_x
        # threshold relative to shoulder width if available
        shoulder_w = abs(left_shoulder[0] - right_shoulder[0]) if left_shoulder and right_shoulder else 100
        if left_dx < -0.4 * shoulder_w:
            labels.append("step_left")
        if right_dx > 0.4 * shoulder_w:
            labels.append("step_right")

    return labels

def smooth_pose_sequence(seq, fill_method="forward"):
    """Simple smoothing: replace None with previous available frame (forward fill)"""
    out = []
    prev = None
    for lm in seq:
        if lm is None:
            if prev is not None:
                out.append(prev)
            else:
                out.append(None)
        else:
            prev = lm
            out.append(lm)
    return out
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle


def classify_pose(landmarks):
    """Return label list based on keypoint angles / positions."""
    labels = []
    if landmarks is None or len(landmarks) < 33:
        return labels

    # Extract key landmarks
    left_shoulder = [landmarks[11].x, landmarks[11].y]
    right_shoulder = [landmarks[12].x, landmarks[12].y]
    left_elbow = [landmarks[13].x, landmarks[13].y]
    right_elbow = [landmarks[14].x, landmarks[14].y]
    left_wrist = [landmarks[15].x, landmarks[15].y]
    right_wrist = [landmarks[16].x, landmarks[16].y]
    left_hip = [landmarks[23].x, landmarks[23].y]
    right_hip = [landmarks[24].x, landmarks[24].y]
    left_knee = [landmarks[25].x, landmarks[25].y]
    right_knee = [landmarks[26].x, landmarks[26].y]

    # Calculate joint angles
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    left_knee_angle = calculate_angle(left_hip, left_knee, [landmarks[27].x, landmarks[27].y])
    right_knee_angle = calculate_angle(right_hip, right_knee, [landmarks[28].x, landmarks[28].y])

    # Pose classification rules
    if left_elbow_angle < 40 and right_elbow_angle < 40:
        labels.append("Floss Arms Crossed")
    elif left_elbow_angle > 150 and right_elbow_angle > 150:
        labels.append("Hands Up")
    elif left_knee_angle < 100 or right_knee_angle < 100:
        labels.append("Squat")
    elif abs(left_shoulder[0] - right_shoulder[0]) > 0.3:
        labels.append("Side Sway")

    # If both arms alternate (floss pattern approx)
    if "Floss Arms Crossed" in labels and (left_knee_angle < 120 or right_knee_angle < 120):
        labels.append("Floss Step")

    if not labels:
        labels.append("Neutral Pose")

    return labels