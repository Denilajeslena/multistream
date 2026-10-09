import os
import cv2
import uuid
import time
import shutil
import logging
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from server.backend.config import settings
from server.backend.db.repository import Repository
from server.backend.ingestion.buffer_manager import buffer_manager

logger = logging.getLogger(__name__)

class EvidenceClipGenerator:
    """
    Generates short video clips (default +/-5 seconds) around matched events.
    Uses FFmpeg if available with OpenCV VideoWriter fallback.
    Only generates clips on-demand.
    """
    def __init__(self):
        self.evidence_dir = settings.EVIDENCE_DIR
        self.ffmpeg_path = shutil.which("ffmpeg")

    def generate_clip(self, camera_id: str, center_timestamp: float, frame_id: str, half_window_sec: float = 5.0) -> Optional[str]:
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        clip_uuid = uuid.uuid4().hex[:8]
        start_time = center_timestamp - half_window_sec
        end_time = center_timestamp + half_window_sec
        clip_filename = f"clip_{camera_id}_{int(center_timestamp)}_{clip_uuid}.mp4"
        clip_filepath = self.evidence_dir / clip_filename
        rel_path = f"data/evidence/{clip_filename}"

        # 1. First attempt to gather frames from buffer_manager (higher temporal resolution)
        buffered_frames = buffer_manager.get_frames_in_window(camera_id, center_timestamp, half_window_sec)

        images: List[np.ndarray] = []
        if buffered_frames:
            for _, b_bytes in buffered_frames:
                arr = np.frombuffer(b_bytes, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    images.append(img)

        # 2. If buffer has fewer than 3 frames (e.g., event occurred minutes ago), load from stored frames
        if len(images) < 3:
            db_frames = Repository.get_frames_around_timestamp(camera_id, center_timestamp, window_sec=half_window_sec)
            for f in db_frames:
                f_full = settings.BASE_DIR / f["frame_path"]
                if f_full.exists():
                    img = cv2.imread(str(f_full))
                    if img is not None:
                        images.append(img)

        # 3. If still empty, check the center frame itself
        if not images:
            center_frame_record = Repository.get_frame(frame_id)
            if center_frame_record:
                f_full = settings.BASE_DIR / center_frame_record["frame_path"]
                if f_full.exists():
                    img = cv2.imread(str(f_full))
                    if img is not None:
                        images = [img] * 10  # Repeat single frame for 2 seconds

        if not images:
            logger.warning(f"No frames available to generate evidence clip for {camera_id} at {center_timestamp}")
            return None

        # Standardize dimensions
        target_w = settings.FRAME_WIDTH
        target_h = settings.FRAME_HEIGHT
        resized_imgs = []
        for img in images:
            if img.shape[1] != target_w or img.shape[0] != target_h:
                resized_imgs.append(cv2.resize(img, (target_w, target_h)))
            else:
                resized_imgs.append(img)

        # Output video FPS calculation
        duration_sec = max(1.0, end_time - start_time)
        fps = max(1.0, round(len(resized_imgs) / duration_sec, 1))

        # Try FFmpeg method if available
        created = False
        if self.ffmpeg_path:
            created = self._create_with_ffmpeg(resized_imgs, fps, clip_filepath)

        if not created:
            created = self._create_with_opencv(resized_imgs, fps, clip_filepath)

        if created and clip_filepath.exists():
            Repository.insert_evidence_clip(
                clip_id=clip_uuid,
                frame_id=frame_id,
                camera_id=camera_id,
                center=center_timestamp,
                start_t=start_time,
                end_t=end_time,
                clip_path=rel_path
            )
            return rel_path
        return None

    def _create_with_ffmpeg(self, images: List[np.ndarray], fps: float, output_path: Path) -> bool:
        try:
            # Stream raw BGR24 frames via stdin pipe to FFmpeg H.264
            h, w = images[0].shape[:2]
            cmd = [
                self.ffmpeg_path,
                "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{w}x{h}",
                "-pix_fmt", "bgr24",
                "-r", str(fps),
                "-i", "-",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "ultrafast",
                "-crf", "26",
                str(output_path)
            ]
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for img in images:
                process.stdin.write(img.tobytes())
            process.stdin.close()
            process.wait(timeout=10)
            return process.returncode == 0 and output_path.exists()
        except Exception as e:
            logger.warning(f"FFmpeg generation failed ({e}), falling back to OpenCV VideoWriter.")
            return False

    def _create_with_opencv(self, images: List[np.ndarray], fps: float, output_path: Path) -> bool:
        try:
            h, w = images[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
            for img in images:
                writer.write(img)
            writer.release()
            return output_path.exists()
        except Exception as e:
            logger.error(f"OpenCV VideoWriter failed: {e}")
            return False

clip_generator = EvidenceClipGenerator()
