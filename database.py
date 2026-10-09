import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from server.backend.config import settings

_local = threading.local()

def get_connection(db_path: Path = None) -> sqlite3.Connection:
    target_path = str(db_path or settings.DB_PATH)
    if not hasattr(_local, "connections"):
        _local.connections = {}
    if target_path not in _local.connections:
        conn = sqlite3.connect(target_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        _local.connections[target_path] = conn
    return _local.connections[target_path]

@contextmanager
def db_session(db_path: Path = None):
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise

def init_db(db_path: Path = None):
    settings.ensure_directories()
    conn = get_connection(db_path)
    with conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS cameras (
            camera_id TEXT PRIMARY KEY,
            camera_name TEXT NOT NULL,
            status TEXT DEFAULT 'OFFLINE',
            last_seen REAL DEFAULT 0,
            fps REAL DEFAULT 0.0,
            frame_count INTEGER DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS camera_aliases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alias TEXT UNIQUE NOT NULL,
            camera_id TEXT NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY (camera_id) REFERENCES cameras (camera_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS frames (
            frame_id TEXT PRIMARY KEY,
            camera_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            timestamp_iso TEXT NOT NULL,
            frame_path TEXT NOT NULL,
            width INTEGER DEFAULT 640,
            height INTEGER DEFAULT 360,
            created_at REAL NOT NULL,
            FOREIGN KEY (camera_id) REFERENCES cameras (camera_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_frames_camera_time ON frames (camera_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_frames_timestamp ON frames (timestamp);

        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_id TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            box_x1 REAL NOT NULL,
            box_y1 REAL NOT NULL,
            box_x2 REAL NOT NULL,
            box_y2 REAL NOT NULL,
            FOREIGN KEY (frame_id) REFERENCES frames (frame_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_detections_label ON detections (label);
        CREATE INDEX IF NOT EXISTS idx_detections_frame ON detections (frame_id);

        CREATE TABLE IF NOT EXISTS evidence_clips (
            clip_id TEXT PRIMARY KEY,
            frame_id TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            timestamp_center REAL NOT NULL,
            timestamp_start REAL NOT NULL,
            timestamp_end REAL NOT NULL,
            clip_path TEXT NOT NULL,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS realtime_analysis_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            event_type TEXT NOT NULL,
            objects TEXT NOT NULL,
            confidence REAL NOT NULL,
            tracking_ids TEXT NOT NULL,
            bounding_box TEXT,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_realtime_events_camera_time
            ON realtime_analysis_events (camera_id, timestamp DESC);
        """)

        # Seed initial camera records & default aliases if not present
        default_cameras = [
            ("CAM-01", "Main Gate"),
            ("CAM-02", "Parking Area"),
            ("CAM-03", "Exit Gate"),
        ]
        import time
        now = time.time()
        for cid, cname in default_cameras:
            conn.execute("""
            INSERT OR IGNORE INTO cameras (camera_id, camera_name, status, last_seen, fps, frame_count, created_at, updated_at)
            VALUES (?, ?, 'OFFLINE', 0, 0.0, 0, ?, ?);
            """, (cid, cname, now, now))

        default_aliases = [
            ("main gate", "CAM-01"),
            ("gate 1", "CAM-01"),
            ("entrance", "CAM-01"),
            ("front gate", "CAM-01"),
            ("parking", "CAM-02"),
            ("parking area", "CAM-02"),
            ("parking lot", "CAM-02"),
            ("car park", "CAM-02"),
            ("exit", "CAM-03"),
            ("exit gate", "CAM-03"),
            ("gate 3", "CAM-03"),
            ("back gate", "CAM-03")
        ]
        for alias, cid in default_aliases:
            conn.execute("""
            INSERT OR IGNORE INTO camera_aliases (alias, camera_id, created_at)
            VALUES (?, ?, ?);
            """, (alias.lower().strip(), cid, now))
