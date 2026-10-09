#!/usr/bin/env python3
"""
Storage cleanup utility.
Removes temporary generated evidence clips and optionally prunes old sampled frames.
Usage:
    python scripts/cleanup.py [--hours 6] [--all-evidence] [--frames-older-than-hours 24]
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from server.backend.config import settings
from server.backend.db.database import get_connection, db_session

def cleanup_evidence(older_than_hours: float = 6.0, delete_all: bool = False):
    evidence_dir = settings.EVIDENCE_DIR
    if not evidence_dir.exists():
        print(f"Evidence directory {evidence_dir} does not exist.")
        return

    cutoff = time.time() - (older_than_hours * 3600.0)
    count = 0
    bytes_freed = 0

    print(f"Cleaning evidence in {evidence_dir} (older than {older_than_hours}h, delete_all={delete_all})...")

    with db_session() as conn:
        for file in evidence_dir.glob("*.mp4"):
            mtime = file.stat().st_mtime
            if delete_all or mtime < cutoff:
                size = file.stat().st_size
                try:
                    file.unlink()
                    count += 1
                    bytes_freed += size
                    # Prune record from database
                    conn.execute("DELETE FROM evidence_clips WHERE clip_path LIKE ?", (f"%{file.name}%",))
                except Exception as e:
                    print(f"Error removing {file.name}: {e}")

    mb = bytes_freed / (1024 * 1024)
    print(f"Evidence cleanup complete: removed {count} files, freed {mb:.2f} MB.")

def cleanup_frames(older_than_hours: float = 24.0):
    frames_dir = settings.FRAMES_DIR
    if not frames_dir.exists():
        return

    cutoff = time.time() - (older_than_hours * 3600.0)
    count = 0
    bytes_freed = 0

    print(f"Pruning frames in {frames_dir} older than {older_than_hours} hours...")

    with db_session() as conn:
        for file in frames_dir.glob("*.jpg"):
            mtime = file.stat().st_mtime
            if mtime < cutoff:
                size = file.stat().st_size
                try:
                    file.unlink()
                    count += 1
                    bytes_freed += size
                    conn.execute("DELETE FROM frames WHERE frame_id = ?", (file.stem,))
                    conn.execute("DELETE FROM detections WHERE frame_id = ?", (file.stem,))
                except Exception as e:
                    print(f"Error removing {file.name}: {e}")

    mb = bytes_freed / (1024 * 1024)
    print(f"Frames cleanup complete: removed {count} files, freed {mb:.2f} MB.")

def main():
    parser = argparse.ArgumentParser(description="Clean up temporary CCTV evidence and old frames.")
    parser.add_argument("--hours", type=float, default=6.0, help="Delete evidence clips older than this many hours (default: 6)")
    parser.add_argument("--all-evidence", action="store_true", help="Delete all generated evidence clips immediately")
    parser.add_argument("--frames-older-than-hours", type=float, default=None, help="Prune sampled frames older than specified hours")
    args = parser.parse_args()

    cleanup_evidence(older_than_hours=args.hours, delete_all=args.all_evidence)
    if args.frames_older_than_hours is not None:
        cleanup_frames(older_than_hours=args.frames_older_than_hours)

if __name__ == "__main__":
    main()
