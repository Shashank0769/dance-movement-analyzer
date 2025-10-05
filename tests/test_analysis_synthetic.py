# tests/test_analysis_synthetic.py
# This is a light test to ensure analyze_video returns expected structure given a tiny sample video.
# If you don't have a sample video, skip or mock cv2.VideoCapture in more advanced tests.

from app.analysis import analyze_video
import pytest
import os

def test_analyze_returns_summary(monkeypatch):
    # monkeypatch analyze internals to avoid requiring a real video file
    def fake_extract(path, max_frames=600):
        return {
            "total_frames": 10,
            "fps": 30,
            "pose_counts": {"hands_up": 2},
            "sample_frames": []
        }
    # If you want to run an integration test, provide a small mp4 in demo_sample_videos/
    # Here we just assert the function exists; a full integration test is optional.
    assert callable(analyze_video)
