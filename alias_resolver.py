import logging
from typing import Optional, Tuple, List, Dict
from server.backend.db.repository import Repository

logger = logging.getLogger(__name__)

class CameraAliasResolver:
    """
    Manages persistent camera memory and location resolution.
    Survives server restart, chat restart, and UI reload via SQLite.
    """
    def resolve(self, location_text: Optional[str]) -> Tuple[Optional[str], bool]:
        """
        Returns:
            (camera_id, is_unresolved_unknown_location)
            - If location is None or empty: (None, False) -> search all cameras
            - If location is recognized: (camera_id, False)
            - If location is specified but unknown: (None, True) -> prompt user
        """
        if not location_text:
            return None, False

        clean_loc = location_text.strip().lower()
        if not clean_loc:
            return None, False

        camera_id = Repository.resolve_alias(clean_loc)
        if camera_id:
            return camera_id, False

        # If user typed something like "camera 1" or "cam 2" or "cam-03"
        for num, cid in [("1", "CAM-01"), ("01", "CAM-01"), ("2", "CAM-02"), ("02", "CAM-02"), ("3", "CAM-03"), ("03", "CAM-03")]:
            if clean_loc in [f"cam {num}", f"cam{num}", f"camera {num}", f"cam-{num}", cid.lower()]:
                return cid, False

        # Unknown location!
        return None, True

    def register_alias(self, alias: str, camera_id: str) -> bool:
        clean_alias = alias.strip().lower()
        clean_cid = camera_id.strip().upper()
        if not clean_alias or not clean_cid:
            return False
        return Repository.add_alias(clean_alias, clean_cid)

    def get_all_mappings(self) -> List[Dict[str, str]]:
        return Repository.get_all_aliases()

alias_resolver = CameraAliasResolver()
