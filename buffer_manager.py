import time
import threading
from collections import deque
from typing import Dict, List, Tuple, Optional
import numpy as np
from server.backend.config import settings

class FrameBufferItem:
    def __init__(self, sequence: int, timestamp: float, frame_bytes: bytes, width: int = 640, height: int = 360):
        self.sequence = sequence
        self.timestamp = timestamp
        self.frame_bytes = frame_bytes
        self.width = width
        self.height = height

class CameraBufferManager:
    """
    Maintains a rolling ring buffer of recent frames in memory per camera.
    Used to quickly generate +/- 5-second evidence clips around events without
    writing every raw frame to disk.
    """
    def __init__(self, max_seconds: int = 30):
        self.max_seconds = max_seconds
        self._buffers: Dict[str, deque] = {
            "CAM-01": deque(),
            "CAM-02": deque(),
            "CAM-03": deque(),
        }
        self._lock = threading.Lock()
        self._sequence = 0

    def add_frame(self, camera_id: str, timestamp: float, frame_bytes: bytes, width: int = 640, height: int = 360):
        with self._lock:
            if camera_id not in self._buffers:
                self._buffers[camera_id] = deque()
            buf = self._buffers[camera_id]
            self._sequence += 1
            buf.append(FrameBufferItem(self._sequence, timestamp, frame_bytes, width, height))

            # Evict frames older than max_seconds
            cutoff = timestamp - self.max_seconds
            while buf and buf[0].timestamp < cutoff:
                buf.popleft()

    def get_frames_in_window(self, camera_id: str, center_timestamp: float, half_window_sec: float = 5.0) -> List[Tuple[float, bytes]]:
        start_time = center_timestamp - half_window_sec
        end_time = center_timestamp + half_window_sec
        with self._lock:
            buf = self._buffers.get(camera_id, deque())
            items = [
                (item.timestamp, item.frame_bytes)
                for item in buf
                if start_time <= item.timestamp <= end_time
            ]
            return items

    def get_latest_frame(self, camera_id: str) -> Optional[bytes]:
        with self._lock:
            buf = self._buffers.get(camera_id, deque())
            if buf:
                return buf[-1].frame_bytes
            return None

    def get_latest_frame_with_sequence(self, camera_id: str) -> Optional[Tuple[int, bytes]]:
        with self._lock:
            buf = self._buffers.get(camera_id, deque())
            if buf:
                latest = buf[-1]
                return latest.sequence, latest.frame_bytes
            return None

    def get_latest_frame_details(self, camera_id: str) -> Optional[Tuple[int, float, bytes]]:
        with self._lock:
            buf = self._buffers.get(camera_id, deque())
            if buf:
                latest = buf[-1]
                return latest.sequence, latest.timestamp, latest.frame_bytes
            return None

buffer_manager = CameraBufferManager(max_seconds=settings.BUFFER_SECONDS)
