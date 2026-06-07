"""
user_manager.py — ID-based user management for VisPay
Stores: user_id, name, pin_hash, enrolled_at
PIN is hashed with SHA-256 (bcrypt-compatible if available).
"""

import os
import json
import time
import hashlib

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR   = os.path.join(BASE_DIR, "enrolled_voices")
INDEX_FILE = os.path.join(SAVE_DIR, "users.json")

# ── optional bcrypt ───────────────────────────────────────────────────────────
try:
    import bcrypt
    _BCRYPT = True
except ImportError:
    _BCRYPT = False


# ══════════════════════════════════════════════════════════════════════════════
# PIN HASHING
# ══════════════════════════════════════════════════════════════════════════════

def _hash_pin(pin: str) -> str:
    if _BCRYPT:
        return bcrypt.hashpw(pin.encode(), bcrypt.gensalt()).decode()
    return hashlib.sha256(f"vispay_{pin}_salt".encode()).hexdigest()


def _check_pin(pin: str, stored_hash: str) -> bool:
    if _BCRYPT:
        try:
            return bcrypt.checkpw(pin.encode(), stored_hash.encode())
        except Exception:
            return False
    return hashlib.sha256(f"vispay_{pin}_salt".encode()).hexdigest() == stored_hash


# ══════════════════════════════════════════════════════════════════════════════
# PIN COLLECTION  (voice + keyboard)
# ══════════════════════════════════════════════════════════════════════════════

def _collect_pin(prompt_voice: str, prompt_kb: str) -> str:
    """
    Ask user for a 4-digit PIN via voice or keyboard.
    Keeps asking until exactly 4 digits are received.
    """
    from tts import speak

    while True:
        speak(prompt_voice, force=True)
        print(f"\n  {prompt_kb}")
        raw = input("  PIN (4 digits): ").strip()

        # Strip spoken words like "my pin is 1234"
        digits = "".join(c for c in raw if c.isdigit())

        if len(digits) == 4:
            return digits

        speak("Please enter exactly 4 digits.", force=True)


def collect_and_confirm_pin() -> str:
    """
    Collect PIN twice and confirm they match.
    Returns the confirmed 4-digit PIN string.
    """
    from tts import speak

    while True:
        pin1 = _collect_pin(
            prompt_voice="Please create a 4-digit PIN for your account.",
            prompt_kb="Create your 4-digit PIN:"
        )
        pin2 = _collect_pin(
            prompt_voice="Please repeat your PIN to confirm.",
            prompt_kb="Confirm your 4-digit PIN:"
        )

        if pin1 == pin2:
            speak("PIN confirmed.", force=True)
            return pin1

        speak("PINs did not match. Please try again.", force=True)
        print("  ✗ PINs did not match. Try again.\n")


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

def assign_new_id(name: str = "", pin_hash: str = "") -> str:
    """
    Assign next available ID: user_001, user_002 ...
    Stores name and pin_hash alongside enrolled_at.
    """
    index  = _load_index()
    n      = len(index) + 1
    user_id = f"user_{n:03d}"
    index[user_id] = {
        "enrolled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "name":        name,
        "pin_hash":    pin_hash,
    }
    _save_index(index)
    return user_id


def get_user(user_id: str) -> dict | None:
    """Return user record dict or None."""
    return _load_index().get(user_id)


def verify_pin(user_id: str, pin: str) -> bool:
    """Return True if PIN matches stored hash for this user."""
    user = get_user(user_id)
    if not user or not user.get("pin_hash"):
        return False
    return _check_pin(pin, user["pin_hash"])


def update_pin(user_id: str, new_pin: str) -> bool:
    """Hash and store a new PIN for user_id. Returns True on success."""
    index = _load_index()
    if user_id not in index:
        return False
    index[user_id]["pin_hash"] = _hash_pin(new_pin)
    _save_index(index)
    return True


def list_enrolled() -> list:
    return list(_load_index().keys())


def enrolled_count() -> int:
    return len(_load_index())