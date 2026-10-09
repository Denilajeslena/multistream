import json
import time
from typing import List, Optional, Dict, Any, Tuple
from server.backend.db.database import get_connection, db_session
from server.backend.db.models import CameraModel, CameraAliasModel, DetectionItem, FrameRecord

class Repository:
    @staticmethod
    def insert_realtime_event(event: Dict[str, Any]) -> int:
        now = time.time()
        with db_session() as conn:
            cursor = conn.execute(
                """
                INSERT INTO realtime_analysis_events
                    (camera_id, timestamp, event_type, objects, confidence,
                     tracking_ids, bounding_box, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event["camera_id"],
                    event["timestamp"],
                    event["event_type"],
                    json.dumps(event["objects"]),
                    event["confidence"],
                    json.dumps(event["tracking_ids"]),
                    json.dumps(event["bounding_box"]) if event.get("bounding_box") else None,
                    now,
                ),
            )
            return int(cursor.lastrowid)

    @staticmethod
    def get_realtime_events(camera_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        conn = get_connection()
        if camera_id:
            rows = conn.execute(
                """
                SELECT * FROM realtime_analysis_events
                WHERE camera_id = ?
                ORDER BY timestamp DESC LIMIT ?
                """,
                (camera_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM realtime_analysis_events
                ORDER BY timestamp DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        events = []
        for row in rows:
            event = dict(row)
            event["objects"] = json.loads(event["objects"])
            event["tracking_ids"] = json.loads(event["tracking_ids"])
            event["bounding_box"] = (
                json.loads(event["bounding_box"]) if event["bounding_box"] else None
            )
            events.append(event)
        return events

    @staticmethod
    def upsert_camera(camera_id: str, camera_name: Optional[str] = None, status: str = "ONLINE", fps: float = 0.0, frame_increment: int = 1):
        now = time.time()
        with db_session() as conn:
            row = conn.execute("SELECT camera_id, camera_name, frame_count FROM cameras WHERE camera_id = ?", (camera_id,)).fetchone()
            if row:
                name = camera_name or row["camera_name"]
                new_count = row["frame_count"] + frame_increment
                conn.execute("""
                UPDATE cameras
                SET camera_name = ?, status = ?, last_seen = ?, fps = ?, frame_count = ?, updated_at = ?
                WHERE camera_id = ?
                """, (name, status, now, fps, new_count, now, camera_id))
            else:
                name = camera_name or camera_id
                conn.execute("""
                INSERT INTO cameras (camera_id, camera_name, status, last_seen, fps, frame_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (camera_id, name, status, now, fps, frame_increment, now, now))

    @staticmethod
    def get_cameras() -> List[Dict[str, Any]]:
        conn = get_connection()
        rows = conn.execute("SELECT * FROM cameras ORDER BY camera_id ASC").fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def get_camera(camera_id: str) -> Optional[Dict[str, Any]]:
        conn = get_connection()
        row = conn.execute("SELECT * FROM cameras WHERE camera_id = ?", (camera_id,)).fetchone()
        return dict(row) if row else None

    @staticmethod
    def update_offline_cameras(timeout_seconds: float = 6.0):
        cutoff = time.time() - timeout_seconds
        with db_session() as conn:
            conn.execute("""
            UPDATE cameras
            SET status = 'OFFLINE', fps = 0.0
            WHERE last_seen < ? AND status = 'ONLINE'
            """, (cutoff,))

    # Frame metadata
    @staticmethod
    def insert_frame(frame_id: str, camera_id: str, timestamp: float, timestamp_iso: str, frame_path: str, width: int = 640, height: int = 360) -> bool:
        now = time.time()
        with db_session() as conn:
            # Prevent duplicate processing after server restart
            existing = conn.execute("SELECT frame_id FROM frames WHERE frame_id = ?", (frame_id,)).fetchone()
            if existing:
                return False
            conn.execute("""
            INSERT INTO frames (frame_id, camera_id, timestamp, timestamp_iso, frame_path, width, height, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (frame_id, camera_id, timestamp, timestamp_iso, frame_path, width, height, now))
            return True

    @staticmethod
    def get_frame(frame_id: str) -> Optional[Dict[str, Any]]:
        conn = get_connection()
        row = conn.execute("SELECT * FROM frames WHERE frame_id = ?", (frame_id,)).fetchone()
        return dict(row) if row else None

    @staticmethod
    def get_frames(camera_id: Optional[str] = None, start_time: Optional[float] = None, end_time: Optional[float] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        conn = get_connection()
        query = "SELECT * FROM frames WHERE 1=1"
        params = []
        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)
        if start_time is not None:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time is not None:
            query += " AND timestamp <= ?"
            params.append(end_time)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def get_frames_around_timestamp(camera_id: str, center_time: float, window_sec: float = 5.0) -> List[Dict[str, Any]]:
        conn = get_connection()
        start_time = center_time - window_sec
        end_time = center_time + window_sec
        rows = conn.execute("""
        SELECT * FROM frames
        WHERE camera_id = ? AND timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp ASC
        """, (camera_id, start_time, end_time)).fetchall()
        return [dict(r) for r in rows]

    # Detections
    @staticmethod
    def insert_detection(frame_id: str, camera_id: str, timestamp: float, label: str, confidence: float, box: List[float]):
        with db_session() as conn:
            conn.execute("""
            INSERT INTO detections (frame_id, camera_id, timestamp, label, confidence, box_x1, box_y1, box_x2, box_y2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (frame_id, camera_id, timestamp, label.lower().strip(), confidence, box[0], box[1], box[2], box[3]))

    @staticmethod
    def get_detections_for_frame(frame_id: str) -> List[Dict[str, Any]]:
        conn = get_connection()
        rows = conn.execute("SELECT * FROM detections WHERE frame_id = ?", (frame_id,)).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def find_detections_by_label(label: str, camera_id: Optional[str] = None, min_confidence: float = 0.15) -> List[Dict[str, Any]]:
        conn = get_connection()
        query = "SELECT * FROM detections WHERE label LIKE ? AND confidence >= ?"
        params = [f"%{label.lower().strip()}%", min_confidence]
        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)
        query += " ORDER BY confidence DESC"
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # Aliases
    @staticmethod
    def get_all_aliases() -> List[Dict[str, Any]]:
        conn = get_connection()
        rows = conn.execute("SELECT * FROM camera_aliases ORDER BY alias ASC").fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def resolve_alias(location_str: str) -> Optional[str]:
        if not location_str:
            return None
        loc = location_str.lower().strip()
        conn = get_connection()
        # Direct match
        row = conn.execute("SELECT camera_id FROM camera_aliases WHERE alias = ?", (loc,)).fetchone()
        if row:
            return row["camera_id"]
        # Check camera_id directly (e.g. "cam-01", "cam01", "cam 1", "cam-1")
        clean_loc = loc.replace(" ", "").replace("_", "").replace("-", "")
        for cid in ["CAM-01", "CAM-02", "CAM-03"]:
            if clean_loc == cid.lower().replace("-", ""):
                return cid
        # Substring / fuzzy match in aliases
        row = conn.execute("SELECT camera_id FROM camera_aliases WHERE ? LIKE '%' || alias || '%' OR alias LIKE '%' || ? || '%'", (loc, loc)).fetchone()
        if row:
            return row["camera_id"]
        return None

    @staticmethod
    def add_alias(alias: str, camera_id: str) -> bool:
        clean_alias = alias.lower().strip()
        now = time.time()
        with db_session() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO camera_aliases (alias, camera_id, created_at)
            VALUES (?, ?, ?)
            """, (clean_alias, camera_id, now))
            return True

    # Evidence clips
    @staticmethod
    def insert_evidence_clip(clip_id: str, frame_id: str, camera_id: str, center: float, start_t: float, end_t: float, clip_path: str):
        now = time.time()
        with db_session() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO evidence_clips (clip_id, frame_id, camera_id, timestamp_center, timestamp_start, timestamp_end, clip_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (clip_id, frame_id, camera_id, center, start_t, end_t, clip_path, now))

    @staticmethod
    def get_evidence_clip(clip_id: str) -> Optional[Dict[str, Any]]:
        conn = get_connection()
        row = conn.execute("SELECT * FROM evidence_clips WHERE clip_id = ?", (clip_id,)).fetchone()
        return dict(row) if row else None
