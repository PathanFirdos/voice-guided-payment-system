"""
user_manager.py — Face enrollment user index for VisPay face_auth.
Mirrors voice_auth/user_manager.py so both systems share the same user IDs.

Stores: user_id, name, enrolled_at, image_count
Index file: enrolled_faces/users.json
"""

import os
import json
import time

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR   = os.path.join(BASE_DIR, "enrolled_faces")
INDEX_FILE = os.path.join(SAVE_DIR, "users.json")


# ══════════════════════════════════════════════════════════════════════════════
# INDEX HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_index() -> dict:
    if not os.path.exists(INDEX_FILE):
        return {}
    with open(INDEX_FILE) as f:
        return json.load(f)


def _save_index(index: dict):
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(INDEX_FILE, "w") as f:
        json.dump(index, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def register_user(user_id: str, name: str = "", image_count: int = 0) -> bool:
    """
    Register or update a face enrollment entry.
    Returns True if newly created, False if updated existing.
    """
    index  = _load_index()
    is_new = user_id not in index
    index[user_id] = {
        "enrolled_at":  time.strftime("%Y-%m-%d %H:%M:%S"),
        "name":         name or user_id,
        "image_count":  image_count,
    }
    _save_index(index)
    return is_new


def get_user(user_id: str) -> dict | None:
    """Return user record or None."""
    return _load_index().get(user_id)


def list_enrolled() -> list:
    """Return list of all enrolled user_ids."""
    return list(_load_index().keys())


def enrolled_count() -> int:
    return len(_load_index())


def remove_user(user_id: str) -> bool:
    """Remove user from index (does not delete image folder)."""
    index = _load_index()
    if user_id not in index:
        return False
    del index[user_id]
    _save_index(index)
    return True


def sync_from_folders() -> dict:
    """
    Scan enrolled_faces/ and auto-register any folder not yet in the index.
    Useful for fixing enrollments that were saved without going through this manager.
    Returns the updated index.
    """
    index = _load_index()
    if not os.path.exists(SAVE_DIR):
        return index

    for folder in os.listdir(SAVE_DIR):
        folder_path = os.path.join(SAVE_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        if folder not in index:
            images = [f for f in os.listdir(folder_path)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            index[folder] = {
                "enrolled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "name":        folder,
                "image_count": len(images),
            }
            print(f"  [face_user_manager] Auto-registered: {folder} ({len(images)} images)")

    _save_index(index)
    return index


def summary() -> str:
    index = _load_index()
    lines = ["=== Face Auth Enrolled Users ==="]
    for uid, data in index.items():
        lines.append(
            f"  {uid:<12} | {data.get('name','?'):<15} | "
            f"{data.get('image_count',0)} images | "
            f"enrolled {data.get('enrolled_at','?')}"
        )
    if not index:
        lines.append("  (none)")
    return "\n".join(lines)


if __name__ == "__main__":
    print("Syncing enrolled_faces/ folders to index...")
    sync_from_folders()
    print(summary())