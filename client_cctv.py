#!/usr/bin/env python3
"""
Lightweight CCTV Client for distributed laptops.
Captures live frames from a webcam or video source and transmits them to the Central Server.
Usage:
    python scripts/client_cctv.py --camera-id CAM-01 --camera-name "Main Gate" --server http://192.168.1.100:8000 --device 0
"""

import sys
import time
import threading
import argparse
import requests
import cv2

def main():
    parser = argparse.ArgumentParser(description="Distributed CCTV Client")
    parser.add_argument("--camera-id", type=str, required=True, help="Camera Identifier (e.g. CAM-01, CAM-02, CAM-03)")
    parser.add_argument("--camera-name", type=str, default=None, help="Human-readable camera name (e.g. 'Main Gate')")
    parser.add_argument("--server", type=str, required=True, help="Central Server URL (e.g. http://192.168.1.50:8000)")
    parser.add_argument("--device", type=str, default="0", help="Webcam device index (0, 1), video file path, or RTSP URL")
    parser.add_argument("--fps", type=float, default=5.0, help="Target transmission FPS (default: 5)")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=360, help="Frame height (default: 360)")
    parser.add_argument("--quality", type=int, default=80, help="JPEG compression quality (default: 80)")
    args = parser.parse_args()

    # Determine video capture source
    device_source = int(args.device) if args.device.isdigit() else args.device
    cap = cv2.VideoCapture(device_source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.device}", file=sys.stderr)
        sys.exit(1)

    endpoint = f"{args.server.rstrip('/')}/api/cameras/frame"
    camera_name = args.camera_name or args.camera_id
    session = requests.Session()
    interval = 1.0 / max(0.5, args.fps)
    stop_capture = threading.Event()
    latest_frame_lock = threading.Lock()
    latest_frame = {"sequence": 0, "frame": None}
    is_file_source = not args.device.isdigit()

    def capture_latest_frame():
        next_capture = time.monotonic()
        while not stop_capture.is_set():
            ret, frame = cap.read()
            if ret:
                with latest_frame_lock:
                    latest_frame["sequence"] += 1
                    latest_frame["frame"] = frame
            elif is_file_source:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                print("Failed to capture frame from camera.", file=sys.stderr)
                stop_capture.wait(0.5)
                next_capture = time.monotonic()
                continue

            next_capture += interval
            delay = next_capture - time.monotonic()
            if delay > 0:
                stop_capture.wait(delay)
            else:
                next_capture = time.monotonic()

    capture_thread = threading.Thread(
        target=capture_latest_frame,
        name=f"cctv-capture-{args.camera_id}",
        daemon=True,
    )
    capture_thread.start()

    print(f"==================================================")
    print(f" CCTV STREAM CLIENT ACTIVE")
    print(f" Camera ID:   {args.camera_id}")
    print(f" Camera Name: {camera_name}")
    print(f" Source:      {args.device}")
    print(f" Target FPS:  {args.fps}")
    print(f" Server:      {endpoint}")
    print(f"==================================================")

    last_sequence = 0
    try:
        while True:
            with latest_frame_lock:
                sequence = latest_frame["sequence"]
                frame = latest_frame["frame"]
            if frame is None or sequence == last_sequence:
                time.sleep(0.005)
                continue

            # Resize to standard resolution
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)

            encoded_ok, buf = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), args.quality],
            )
            if not encoded_ok:
                print("Failed to encode captured frame.", file=sys.stderr)
                last_sequence = sequence
                continue

            files = {
                "frame": ("frame.jpg", buf.tobytes(), "image/jpeg")
            }
            now = time.time()
            data = {
                "camera_id": args.camera_id,
                "camera_name": camera_name,
                "timestamp": str(now)
            }

            try:
                resp = session.post(endpoint, data=data, files=files, timeout=2.0)
                if resp.status_code != 200:
                    print(f"Server response {resp.status_code}: {resp.text}")
            except Exception as e:
                print(f"Transmission error: {e}")
            finally:
                last_sequence = sequence
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopping CCTV stream client...")
    finally:
        stop_capture.set()
        capture_thread.join(timeout=2.0)
        cap.release()
        print("CCTV client stopped.")

if __name__ == "__main__":
    main()
