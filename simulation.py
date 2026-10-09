import time
import cv2
import threading
import numpy as np
from typing import Dict, Any, Optional
from fastapi import APIRouter, BackgroundTasks

from server.backend.config import settings
from server.backend.ingestion.camera_registry import camera_registry
from server.backend.ingestion.buffer_manager import buffer_manager
from server.backend.ingestion.frame_sampler import frame_sampler
from server.backend.routers.cameras import _ai_indexing_worker

router = APIRouter(prefix="/api/simulation", tags=["Simulation"])

class SyntheticStreamSimulator:
    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.frame_step = 0

    def start(self, fps: float = 2.0):
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._run_loop, args=(fps,), daemon=True)
            self._thread.start()

    def stop(self):
        with self._lock:
            self._running = False

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def _generate_frame(self, camera_id: str, camera_name: str, step: int) -> np.ndarray:
        w = settings.FRAME_WIDTH
        h = settings.FRAME_HEIGHT
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Background environment
        if camera_id == "CAM-01":
            # Main Gate: dark asphalt road + green grass sides + gate booth
            img[:] = (50, 80, 50)  # grass
            cv2.rectangle(img, (0, 80), (w, 280), (70, 70, 70), -1)  # road
            cv2.line(img, (0, 180), (w, 180), (200, 200, 200), 2)  # road divider
            # Gate barrier
            cv2.rectangle(img, (150, 40), (200, 100), (40, 40, 150), -1)
            cv2.line(img, (175, 100), (175, 260), (0, 255, 255), 4)

            # Red car moving across gate
            car_x = int((step * 15) % (w + 120)) - 60
            cv2.rectangle(img, (car_x, 150), (car_x + 90, 210), (0, 0, 220), -1)  # Red body
            cv2.rectangle(img, (car_x + 15, 130), (car_x + 70, 150), (30, 30, 180), -1)  # Roof
            cv2.circle(img, (car_x + 20, 210), 12, (20, 20, 20), -1)  # Wheel
            cv2.circle(img, (car_x + 70, 210), 12, (20, 20, 20), -1)  # Wheel
            cv2.putText(img, "RED CAR", (car_x + 5, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Person with a large bag standing near booth
            px, py = 230, 110
            cv2.circle(img, (px, py), 10, (200, 180, 160), -1)  # head
            cv2.rectangle(img, (px - 8, py + 10), (px + 8, py + 40), (180, 50, 50), -1)  # body
            cv2.rectangle(img, (px + 10, py + 25), (px + 25, py + 45), (10, 10, 10), -1)  # large bag
            cv2.putText(img, "LARGE BAG", (px + 5, py + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        elif camera_id == "CAM-02":
            # Parking Area: gray pavement + parking slots
            img[:] = (60, 60, 60)
            for x in range(50, w, 110):
                cv2.line(img, (x, 50), (x, 300), (255, 255, 255), 2)
            # Parked white truck
            cv2.rectangle(img, (70, 90), (140, 240), (240, 240, 240), -1)
            cv2.putText(img, "WHITE TRUCK", (72, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

            # Walking person in blue shirt
            px = int(250 + 60 * np.sin(step * 0.2))
            py = 180
            cv2.circle(img, (px, py), 10, (200, 180, 160), -1)  # head
            cv2.rectangle(img, (px - 8, py + 10), (px + 8, py + 38), (220, 80, 20), -1)  # blue shirt (BGR)
            cv2.putText(img, "PERSON (BLUE SHIRT)", (px - 30, py - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        else:
            # CAM-03: Exit Gate
            img[:] = (45, 45, 45)
            cv2.rectangle(img, (0, 100), (w, 260), (75, 75, 75), -1)
            cv2.putText(img, "EXIT GATE", (w // 2 - 50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # White truck exiting
            tx = int((step * 12) % (w + 140)) - 80
            cv2.rectangle(img, (tx, 140), (tx + 120, 220), (245, 245, 245), -1)
            cv2.putText(img, "WHITE TRUCK", (tx + 10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # Overlay Camera ID, Name, and Timestamp
        now_ts = time.time()
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_ts))
        cv2.putText(img, f"{camera_id} - {camera_name}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(img, time_str, (w - 210, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return img

    def _run_loop(self, fps: float):
        interval = 1.0 / max(0.5, fps)
        cameras = [
            ("CAM-01", "Main Gate"),
            ("CAM-02", "Parking Area"),
            ("CAM-03", "Exit Gate"),
        ]

        while self.is_running():
            self.frame_step += 1
            now = time.time()

            for cid, cname in cameras:
                frame_img = self._generate_frame(cid, cname, self.frame_step)
                _, encoded = cv2.imencode(".jpg", frame_img, [int(cv2.IMWRITE_JPEG_QUALITY), settings.JPEG_QUALITY])
                frame_bytes = encoded.tobytes()

                # Process through ingestion
                camera_registry.register_frame(cid, now, cname)
                buffer_manager.add_frame(cid, now, frame_bytes)
                frame_sampler.process_and_save_frame(
                    camera_id=cid,
                    timestamp=now,
                    frame_bytes=frame_bytes,
                    ai_pipeline_callback=_ai_indexing_worker
                )

            time.sleep(interval)

simulator = SyntheticStreamSimulator()

@router.post("/start")
def start_simulation(fps: float = 2.0):
    simulator.start(fps=fps)
    return {"status": "started", "fps": fps, "cameras": ["CAM-01", "CAM-02", "CAM-03"]}

@router.post("/stop")
def stop_simulation():
    simulator.stop()
    return {"status": "stopped"}

@router.get("/status")
def simulation_status():
    return {"running": simulator.is_running(), "frames_generated": simulator.frame_step}
