import time
import os
import cv2
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Tuple
from server.backend.config import settings
from server.backend.db.repository import Repository

class FrameSampler:
    """
    Downsamples incoming streams to configurable FPS (default 1 FPS).
    Saves sampled frames to data/frames/ and persists metadata in SQLite.
    Prevents duplicate processing.
    """
    def __init__(self, sampling_fps: float = None):
        self.sampling_interval = 1.0 / (sampling_fps or settings.SAMPLING_FPS)
        self._last_sampled_times: Dict[str, float] = {}

    def should_sample(self, camera_id: str, timestamp: float) -> bool:
        last_t = self._last_sampled_times.get(camera_id, 0.0)
        # Check if enough time has passed based on interval
        if (timestamp - last_t) >= (self.sampling_interval - 0.05):
            self._last_sampled_times[camera_id] = timestamp
            return True
        return False

    def process_and_save_frame(
        self,
        camera_id: str,
        timestamp: float,
        frame_bytes: bytes,
        ai_pipeline_callback = None
    ) -> Optional[Dict[str, Any]]:
        if not self.should_sample(camera_id, timestamp):
            return None

        # Stable unique frame_id
        ts_ms = int(timestamp * 1000)
        frame_id = f"{camera_id}_{ts_ms}"

        # Prevent duplicate processing after server restart
        if Repository.get_frame(frame_id):
            return None

        # Decode image
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return None

        # Resize to standard resolution (e.g. 640x360)
        target_w = settings.FRAME_WIDTH
        target_h = settings.FRAME_HEIGHT
        if img.shape[1] != target_w or img.shape[0] != target_h:
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Write to disk with JPEG compression
        settings.FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{frame_id}.jpg"
        filepath = settings.FRAMES_DIR / filename
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), settings.JPEG_QUALITY]
        cv2.imwrite(str(filepath), img, encode_params)

        iso_timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        rel_path = f"data/frames/{filename}"

        # Store in SQLite metadata
        inserted = Repository.insert_frame(
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp=timestamp,
            timestamp_iso=iso_timestamp,
            frame_path=rel_path,
            width=target_w,
            height=target_h
        )

        frame_info = {
            "frame_id": frame_id,
            "camera_id": camera_id,
            "timestamp": timestamp,
            "timestamp_iso": iso_timestamp,
            "frame_path": rel_path,
            "full_path": str(filepath),
            "image": img
        }

        # Hand off to AI pipeline callback if provided
        if ai_pipeline_callback and inserted:
            ai_pipeline_callback(frame_info)

        return frame_info

frame_sampler = FrameSampler()
