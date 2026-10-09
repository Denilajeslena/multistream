import time
import threading
from typing import Dict, List, Optional, Any
from server.backend.config import settings
from server.backend.db.repository import Repository
from server.backend.db.models import CameraModel

DEFAULT_CAMERAS = {
    "CAM-01": "Main Gate",
    "CAM-02": "Parking Area",
    "CAM-03": "Exit Gate"
}

class CameraState:
    def __init__(self, camera_id: str, camera_name: str):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.status = "OFFLINE"
        self.last_seen = 0.0
        self.fps = 0.0
        self.frame_count = 0
        self._last_frames_timestamps = []
        self._lock = threading.Lock()

    def record_frame(self, timestamp: Optional[float] = None, camera_name: Optional[str] = None):
        receipt_time = time.time()
        with self._lock:
            if camera_name and camera_name.strip():
                self.camera_name = camera_name.strip()
            self.last_seen = receipt_time
            self.status = "ONLINE"
            self.frame_count += 1
            self._last_frames_timestamps.append(receipt_time)

            # Keep only timestamps within last 3 seconds to calculate real-time FPS
            cutoff = receipt_time - 3.0
            self._last_frames_timestamps = [t for t in self._last_frames_timestamps if t >= cutoff]
            if len(self._last_frames_timestamps) > 1:
                duration = self._last_frames_timestamps[-1] - self._last_frames_timestamps[0]
                if duration > 0.1:
                    self.fps = round((len(self._last_frames_timestamps) - 1) / duration, 1)
                else:
                    self.fps = float(len(self._last_frames_timestamps))
            else:
                self.fps = 1.0

    def check_heartbeat(self, timeout_sec: float):
        now = time.time()
        with self._lock:
            if self.status == "ONLINE" and (now - self.last_seen) > timeout_sec:
                self.status = "OFFLINE"
                self.fps = 0.0

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "camera_id": self.camera_id,
                "camera_name": self.camera_name,
                "status": self.status,
                "last_seen": self.last_seen,
                "fps": self.fps,
                "frame_count": self.frame_count,
            }

class CameraRegistry:
    def __init__(self):
        self._cameras: Dict[str, CameraState] = {}
        self._lock = threading.Lock()
        self._init_defaults()

    def _init_defaults(self):
        for cid, cname in DEFAULT_CAMERAS.items():
            self._cameras[cid] = CameraState(cid, cname)

    def register_frame(self, camera_id: str, timestamp: float, camera_name: Optional[str] = None) -> CameraState:
        with self._lock:
            if camera_id not in self._cameras:
                name = camera_name or DEFAULT_CAMERAS.get(camera_id, f"Camera {camera_id}")
                self._cameras[camera_id] = CameraState(camera_id, name)
            cam = self._cameras[camera_id]

        cam.record_frame(timestamp, camera_name)
        # Asynchronously update DB record
        Repository.upsert_camera(camera_id, cam.camera_name, cam.status, cam.fps, frame_increment=1)
        return cam

    def get_camera(self, camera_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cam = self._cameras.get(camera_id)
            if cam:
                cam.check_heartbeat(settings.CAMERA_OFFLINE_TIMEOUT_SEC)
                return cam.to_dict()
        return None

    def get_all_cameras(self) -> List[Dict[str, Any]]:
        with self._lock:
            # Ensure default cameras are always returned
            results = []
            for cid in ["CAM-01", "CAM-02", "CAM-03"]:
                if cid not in self._cameras:
                    self._cameras[cid] = CameraState(cid, DEFAULT_CAMERAS.get(cid, cid))
                cam = self._cameras[cid]
                cam.check_heartbeat(settings.CAMERA_OFFLINE_TIMEOUT_SEC)
                results.append(cam.to_dict())

            for cid, cam in self._cameras.items():
                if cid not in ["CAM-01", "CAM-02", "CAM-03"]:
                    cam.check_heartbeat(settings.CAMERA_OFFLINE_TIMEOUT_SEC)
                    results.append(cam.to_dict())
            return results

    def check_all_heartbeats(self):
        with self._lock:
            for cam in self._cameras.values():
                cam.check_heartbeat(settings.CAMERA_OFFLINE_TIMEOUT_SEC)
        Repository.update_offline_cameras(settings.CAMERA_OFFLINE_TIMEOUT_SEC)

camera_registry = CameraRegistry()
