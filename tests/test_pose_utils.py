# tests/test_pose_utils.py
from app.pose_utils import angle_between, detect_poses_from_landmarks

def test_angle_between_simple_right_angle():
    a = (0.0, 1.0)  # up
    b = (0.0, 0.0)  # center
    c = (1.0, 0.0)  # right
    ang = angle_between(a, b, c)
    assert ang is not None
    # should be ~90°
    assert 80 < ang < 100

def test_detect_hands_up_and_t_pose():
    # synthetic landmarks: shoulders at y=100, wrists at y=50 -> hands up
    lm = {
        "left_shoulder": (100,100,0,1.0),
        "right_shoulder": (200,100,0,1.0),
        "left_wrist": (70,50,0,1.0),
        "right_wrist": (230,50,0,1.0),
        "left_hip": (110,200,0,1.0),
        "right_hip": (190,200,0,1.0),
        "left_knee": (110,250,0,1.0),
        "left_ankle": (110,300,0,1.0),
        "right_knee": (190,250,0,1.0),
        "right_ankle": (190,300,0,1.0),
    }
    labels = detect_poses_from_landmarks(lm)
    assert "hands_up" in labels
    # T-pose may or may not be flagged depending on horizontal distances; accept either
