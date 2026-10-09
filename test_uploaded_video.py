import sys
import threading
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from server.backend.config import settings
from server.backend.db.repository import Repository
from server.backend.main import app
from server.backend.models_ai.interfaces import DetectionResult


def wait_for_upload_status(test_client, job_id, expected_status):
    for _ in range(500):
        response = test_client.get(f"/api/uploads/{job_id}")
        if response.json()["status"] == expected_status:
            return response.json()
        time.sleep(0.01)
    raise AssertionError(f"Uploaded video job did not reach {expected_status}.")


class FakeVideoCapture:
    def __init__(self, _video_path):
        self.frame_index = 0

    def isOpened(self):
        return True

    def get(self, prop):
        import cv2

        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return 90
        if prop == cv2.CAP_PROP_POS_MSEC:
            return max(0, self.frame_index - 1) * 1000 / 30
        return 0

    def grab(self):
        if self.frame_index >= 90:
            return False
        self.frame_index += 1
        return True

    def retrieve(self):
        shade = min(255, (self.frame_index // 30) * 100)
        return True, np.full((360, 640, 3), shade, dtype=np.uint8)

    def release(self):
        pass


def test_uploaded_video_query_returns_timestamp_evidence_and_notifies(monkeypatch, tmp_path):
    from server.backend.ingestion import uploaded_video
    from server.backend.routers import uploads as uploads_router

    monkeypatch.setattr(settings, "UPLOADS_DIR", tmp_path)
    monkeypatch.setattr(uploaded_video.cv2, "VideoCapture", FakeVideoCapture)
    detection_calls = []

    def detect(_image, labels, threshold):
        detection_calls.append((labels, threshold))
        return [DetectionResult(
            label="person using a mobile phone",
            confidence=0.9 + float(_image[0, 0, 0]) / 10000,
            box=[10.0, 20.0, 100.0, 200.0],
        )]

    monkeypatch.setattr(uploaded_video.detector, "detect", detect)
    notifications = []

    async def capture_notification(payload):
        notifications.append(payload)

    monkeypatch.setattr(uploads_router, "send_answer_notification", capture_notification)
    frames_before = Repository.get_frames(limit=100000)

    with TestClient(app) as test_client:
        response = test_client.post(
            "/api/uploads/videos",
            files={"video": ("footage.mp4", b"video bytes", "video/mp4")},
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        status = wait_for_upload_status(test_client, job_id, "completed")
        assert status["frame_count"] == 3

        result_response = test_client.post(
            f"/api/uploads/{job_id}/query",
            json={"question": "Did anyone use a mobile phone?"},
        )
        assert result_response.status_code == 200
        result = result_response.json()
        assert result["matched"] is True
        assert result["timestamp_iso"] == "00:02"
        assert result["confidence"] == 0.92
        assert result["frame_url"].endswith(".jpg")
        assert "a person using a mobile phone" in result["explanation"]
        assert test_client.get(result["frame_url"]).status_code == 200

    assert len(detection_calls) == 3
    cached_result = uploaded_video.query_video(job_id, "Did anyone use a mobile phone?")
    assert cached_result == result
    assert len(detection_calls) == 3
    assert len(Repository.get_frames(limit=100000)) == len(frames_before)
    completed_job = uploaded_video.get_job(job_id)
    assert completed_job is not None
    assert not completed_job["video_path"].exists()
    assert notifications == [{
        "source": "uploaded_video",
        "question": "Did anyone use a mobile phone?",
        "answer": "Yes, a person using a mobile phone was detected at 00:02.",
        "camera": "Uploaded Video",
        "timestamp": "00:02",
        "confidence": 0.92,
        "relevance": 0.92,
        "evidence_frame": result["frame_url"],
        "detections": result["detections"],
        "status": "answer_found",
    }]
    from server.backend.notifications.n8n_webhook import _format_telegram_message

    telegram_text = _format_telegram_message(notifications[0])
    assert "Uploaded Video · 00:02" in telegram_text
    assert "Q: Did anyone use a mobile phone?" in telegram_text
    assert "Confidence 92% · Relevance 92%" in telegram_text
    assert f"Frame: {result['frame_url']}" in telegram_text


def test_uploaded_video_without_matching_evidence_does_not_notify(monkeypatch, tmp_path):
    from server.backend.ingestion import uploaded_video
    from server.backend.routers import uploads as uploads_router

    monkeypatch.setattr(settings, "UPLOADS_DIR", tmp_path)
    monkeypatch.setattr(uploaded_video.cv2, "VideoCapture", FakeVideoCapture)
    monkeypatch.setattr(uploaded_video.detector, "detect", lambda *_args, **_kwargs: [])
    notifications = []

    async def capture_notification(payload):
        notifications.append(payload)

    monkeypatch.setattr(uploads_router, "send_answer_notification", capture_notification)

    with TestClient(app) as test_client:
        upload = test_client.post(
            "/api/uploads/videos",
            files={"video": ("empty-scene.mp4", b"video bytes", "video/mp4")},
        )
        job_id = upload.json()["job_id"]
        wait_for_upload_status(test_client, job_id, "completed")
        result = test_client.post(
            f"/api/uploads/{job_id}/query",
            json={"question": "Did anyone use a mobile phone?"},
        )

    assert upload.status_code == 202
    assert result.status_code == 200
    assert result.json()["matched"] is False
    assert notifications == []


def test_uploaded_video_rejects_unsupported_file_type():
    with TestClient(app) as test_client:
        response = test_client.post(
            "/api/uploads/videos",
            files={"video": ("notes.txt", b"not a video", "text/plain")},
        )

    assert response.status_code == 400


def test_failed_upload_processing_does_not_notify(monkeypatch, tmp_path):
    from server.backend.ingestion import uploaded_video
    from server.backend.routers import uploads as uploads_router

    class UnreadableVideo(FakeVideoCapture):
        def isOpened(self):
            return False

    monkeypatch.setattr(settings, "UPLOADS_DIR", tmp_path)
    monkeypatch.setattr(uploaded_video.cv2, "VideoCapture", UnreadableVideo)
    notifications = []

    async def capture_notification(payload):
        notifications.append(payload)

    monkeypatch.setattr(uploads_router, "send_answer_notification", capture_notification)

    with TestClient(app) as test_client:
        upload = test_client.post(
            "/api/uploads/videos",
            files={"video": ("broken.mp4", b"invalid video", "video/mp4")},
        )
        job_id = upload.json()["job_id"]
        status = wait_for_upload_status(test_client, job_id, "failed")
        query = test_client.post(
            f"/api/uploads/{job_id}/query",
            json={"question": "Did anyone use a mobile phone?"},
        )

    assert upload.status_code == 202
    assert status["status"] == "failed"
    assert query.status_code == 422
    assert notifications == []


def test_uploaded_video_skips_identical_sampled_frames(monkeypatch, tmp_path):
    from server.backend.ingestion import uploaded_video

    class RepeatedVideo(FakeVideoCapture):
        def retrieve(self):
            return True, np.zeros((360, 640, 3), dtype=np.uint8)

    monkeypatch.setattr(settings, "UPLOADS_DIR", tmp_path)
    monkeypatch.setattr(uploaded_video.cv2, "VideoCapture", RepeatedVideo)
    job_dir = tmp_path / "dedupe-job"
    job_dir.mkdir()
    source = job_dir / "source.mp4"
    source.write_bytes(b"test")
    uploaded_video.create_job("dedupe-job", "same-frames.mp4", source, job_dir)

    uploaded_video.process_video("dedupe-job")

    job = uploaded_video.get_job("dedupe-job")
    assert job is not None
    assert job["status"] == "completed"
    assert job["frame_count"] == 1
    assert not source.exists()


def test_uploaded_video_processing_queue_is_bounded():
    from server.backend.ingestion.uploaded_video import (
        MAX_QUEUED_VIDEO_JOBS,
        release_processing_slot,
        reserve_processing_slot,
    )

    reserved = []
    for _ in range(MAX_QUEUED_VIDEO_JOBS + 1):
        reserved.append(reserve_processing_slot())
    try:
        assert all(reserved)
        assert reserve_processing_slot() is False
    finally:
        for _ in reserved:
            release_processing_slot()


def test_uploaded_video_processing_does_not_block_live_health_route(monkeypatch, tmp_path):
    from server.backend.ingestion import uploaded_video

    started = threading.Event()
    continue_processing = threading.Event()

    class SlowVideo(FakeVideoCapture):
        def grab(self):
            started.set()
            if not continue_processing.wait(timeout=5):
                return False
            return super().grab()

    monkeypatch.setattr(settings, "UPLOADS_DIR", tmp_path)
    monkeypatch.setattr(uploaded_video.cv2, "VideoCapture", SlowVideo)
    monkeypatch.setattr(uploaded_video.detector, "detect", lambda *_args, **_kwargs: [])

    try:
        with TestClient(app) as test_client:
            upload = test_client.post(
                "/api/uploads/videos",
                files={"video": ("slow.mp4", b"video bytes", "video/mp4")},
            )
            assert upload.status_code == 202
            job_id = upload.json()["job_id"]
            assert started.wait(timeout=2)

            health = test_client.get("/health")
            assert health.status_code == 200
            continue_processing.set()
            assert wait_for_upload_status(test_client, job_id, "completed")["status"] == "completed"
    finally:
        continue_processing.set()
