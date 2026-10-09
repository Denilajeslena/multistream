#!/usr/bin/env python3
"""
Multi-Camera CCTV Simulation Client.
Simulates CAM-01 (Main Gate), CAM-02 (Parking Area), CAM-03 (Exit Gate)
transmitting video frames over HTTP to the Central Server.
Usage:
    python scripts/simulate_cctv.py [--server http://127.0.0.1:8000] [--fps 2.0] [--duration 60]
"""

import sys
import time
import argparse
import threading
import requests
import cv2
import numpy as np

def generate_scene_frame(camera_id: str, camera_name: str, step: int, width: int = 640, height: int = 360) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)

    if camera_id == "CAM-01":
        # Main Gate
        img[:] = (45, 75, 45)  # grass
        cv2.rectangle(img, (0, 80), (width, 280), (70, 70, 70), -1)  # road
        cv2.line(img, (0, 180), (width, 180), (220, 220, 220), 2)
        # Gate barrier
        cv2.rectangle(img, (140, 40), (190, 100), (30, 30, 160), -1)
        cv2.line(img, (165, 100), (165, 260), (0, 255, 255), 4)

        # Red car passing through
        car_x = int((step * 14) % (width + 140)) - 70
        cv2.rectangle(img, (car_x, 150), (car_x + 90, 210), (0, 0, 220), -1)  # Red body
        cv2.rectangle(img, (car_x + 15, 130), (car_x + 70, 150), (20, 20, 170), -1)  # Cabin
        cv2.circle(img, (car_x + 20, 210), 12, (20, 20, 20), -1)
        cv2.circle(img, (car_x + 70, 210), 12, (20, 20, 20), -1)
        cv2.putText(img, "RED CAR", (car_x + 5, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Large bag near guard booth
        px, py = 220, 110
        cv2.circle(img, (px, py), 10, (200, 180, 160), -1)
        cv2.rectangle(img, (px - 8, py + 10), (px + 8, py + 40), (180, 50, 50), -1)
        cv2.rectangle(img, (px + 10, py + 25), (px + 28, py + 48), (10, 10, 10), -1)
        cv2.putText(img, "LARGE BAG", (px - 5, py + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    elif camera_id == "CAM-02":
        # Parking Area
        img[:] = (60, 60, 60)
        for x in range(40, width, 110):
            cv2.line(img, (x, 50), (x, 310), (255, 255, 255), 2)
        # Parked white truck
        cv2.rectangle(img, (60, 90), (140, 250), (240, 240, 240), -1)
        cv2.putText(img, "WHITE TRUCK", (62, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        # Person with blue shirt
        px = int(240 + 50 * np.sin(step * 0.25))
        py = 180
        cv2.circle(img, (px, py), 10, (200, 180, 160), -1)
        cv2.rectangle(img, (px - 8, py + 10), (px + 8, py + 38), (220, 80, 20), -1)  # Blue shirt
        cv2.putText(img, "PERSON (BLUE SHIRT)", (px - 40, py - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    else:
        # CAM-03: Exit Gate
        img[:] = (50, 50, 50)
        cv2.rectangle(img, (0, 100), (width, 260), (75, 75, 75), -1)
        cv2.putText(img, "EXIT GATE", (width // 2 - 50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # White truck exiting
        tx = int((step * 12) % (width + 140)) - 80
        cv2.rectangle(img, (tx, 140), (tx + 120, 220), (245, 245, 245), -1)
        cv2.putText(img, "WHITE TRUCK", (tx + 10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    now_ts = time.time()
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_ts))
    cv2.putText(img, f"{camera_id} - {camera_name}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(img, time_str, (width - 210, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return img

def run_camera_client(server_url: str, camera_id: str, camera_name: str, fps: float, stop_event: threading.Event):
    endpoint = f"{server_url.rstrip('/')}/api/cameras/frame"
    interval = 1.0 / max(0.5, fps)
    step = 0

    print(f"[{camera_id}] Starting stream to {endpoint} ({camera_name}, {fps} FPS)...")
    session = requests.Session()

    while not stop_event.is_set():
        step += 1
        now = time.time()
        frame = generate_scene_frame(camera_id, camera_name, step)

        _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        files = {
            "frame": ("frame.jpg", buf.tobytes(), "image/jpeg")
        }
        data = {
            "camera_id": camera_id,
            "camera_name": camera_name,
            "timestamp": str(now)
        }

        try:
            resp = session.post(endpoint, data=data, files=files, timeout=2.0)
            if resp.status_code != 200:
                print(f"[{camera_id}] Server returned {resp.status_code}")
        except Exception as e:
            print(f"[{camera_id}] Connection error: {e}")

        time.sleep(interval)

def main():
    parser = argparse.ArgumentParser(description="Multi-Camera CCTV Simulation Client")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000", help="Central Server URL")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second per camera")
    parser.add_argument("--duration", type=float, default=None, help="Run duration in seconds (default: infinite)")
    args = parser.parse_args()

    cameras = [
        ("CAM-01", "Main Gate"),
        ("CAM-02", "Parking Area"),
        ("CAM-03", "Exit Gate")
    ]

    stop_event = threading.Event()
    threads = []
    for cid, cname in cameras:
        t = threading.Thread(target=run_camera_client, args=(args.server, cid, cname, args.fps, stop_event))
        t.daemon = True
        threads.append(t)
        t.start()

    print(f"All 3 simulated cameras (CAM-01, CAM-02, CAM-03) are streaming to {args.server}.")
    print("Press Ctrl+C to terminate simulation.")

    try:
        if args.duration:
            time.sleep(args.duration)
            stop_event.set()
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping simulation...")
        stop_event.set()

    for t in threads:
        t.join(timeout=1.0)
    print("Simulation stopped.")

if __name__ == "__main__":
    main()
