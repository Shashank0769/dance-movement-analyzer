# app/analysis.py
import cv2
import mediapipe as mp
import numpy as np
from collections import Counter
from app.pose_utils import (
    extract_landmarks,
    smooth_pose_sequence,
    classify_pose
)

mp_pose = mp.solutions.pose


def analyze_video(path, max_frames=600, min_confidence=0.5):
    """
    Reads a video, extracts pose keypoints using MediaPipe, applies simple
    classification rules, and returns a JSON summary.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    # Initialize MediaPipe Pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )

    frames = 0
    landmarks_seq = []
    timestamps = []
    per_frame_labels = []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # 🔁 Process frames
    while frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # If no pose detected
        if not results.pose_landmarks:
            landmarks_seq.append(None)
            per_frame_labels.append([])
            timestamps.append(frames / fps)
            frames += 1
            continue

        # 🟢 Extract landmarks
        lm = extract_landmarks(results.pose_landmarks, frame.shape)
        landmarks_seq.append(lm)
        timestamps.append(frames / fps)

        # 🟢 Classify current frame’s pose
        labels = classify_pose(results.pose_landmarks.landmark)
        per_frame_labels.append(labels)

        frames += 1

    cap.release()
    pose.close()

    # 🧹 Smooth missing frames (optional)
    smoothed = smooth_pose_sequence(landmarks_seq)

    # 📊 Aggregate summary
    flat_labels = [label for labels in per_frame_labels for label in labels]
    counts = Counter(flat_labels)
    total_frames = len(per_frame_labels)

    summary = {
        "total_frames": total_frames,
        "fps": fps,
        "pose_counts": dict(counts),
        "sample_frames": [
            {"time_s": timestamps[i], "labels": per_frame_labels[i]}
            for i in range(0, total_frames, max(1, total_frames // 5))
        ],
    }

    return summary
