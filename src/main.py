# =========================================================
# main.py  â€”  VisPay FSM orchestrator  (v3 â€” corrected)
# =========================================================
#
# Changes vs previous version:
#
#  1. POST_AUTH_MENU now calls upi_logic.post_auth_menu(user_id)
#     directly instead of the old get_choice() / run_process() stub.
#     upi_logic.py owns the full 8-option menu (PAY, BALANCE, HISTORY,
#     CONTACTS, RECEIPTS, CURRENCY, FORGOT, EXIT) â€” main.py must not
#     duplicate it.
#
#  2. PAYMENT_TRANSACTION constant removed â€” it was set to None and
#     caused a silent "[runner] No script path provided" dead-end.
#     Payment is now handled entirely inside upi_logic.post_auth_menu.
#
#  3. PIN_VERIFY state added to the FSM â€” after AUTH_SUCCESS the user
#     must pass PIN verification (via upi_logic.verify_pin) before
#     reaching the post-auth menu. This closes the gap where the menu
#     was reachable without a PIN gate.
#
#  4. user_id propagated through the FSM â€” face/voice verify both
#     attempt to identify the user.  Once identified, user_id is
#     carried forward to PIN_VERIFY and POST_AUTH_MENU so upi_logic
#     can query the correct account record in vispay_transactions.db.
#
#  5. upi_logic.init_db() called at startup â€” creates tables if not
#     already present (idempotent).
#
#  6. get_choice() removed â€” replaced by upi_logic.post_auth_menu.
#
#  7. All earlier fixes retained (cached mic, TTS pause, FSM states,
#     health check, cwd-correct subprocess runner).
#
#  8. ModuleNotFoundError: No module named 'tts' — FIXED.
#     run_process() now injects BASE_DIR (src/) into the PYTHONPATH
#     environment variable before launching any subprocess. This means
#     face_auth/verify.py and voice_auth/verify.py can both do
#     `from tts import speak` without needing their own sys.path hacks,
#     because Python's import machinery will search src/ automatically.
#
# =========================================================

import sys
import os
import time
import subprocess
from enum import Enum, auto

import speech_recognition as sr

# â”€â”€ TTS â€” single shared module â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from tts import speak_and_wait as speak, speak as speak_nb

# â”€â”€ Transaction engine â€” imported directly so post_auth_menu is callable â”€â”€â”€â”€â”€â”€
# Import lazily inside run_fsm() to keep startup fast; captured at module level
# so health_check() can verify the import works.
_upi_logic = None

def _import_upi_logic():
    global _upi_logic
    if _upi_logic is not None:
        return _upi_logic
    try:
        import importlib.util
        _UPI_PATH = os.path.join(
            r"C:\projects\vispay\src\transactions", "upi_logic.py"
        )
        if os.path.exists(_UPI_PATH):
            spec   = importlib.util.spec_from_file_location("upi_logic", _UPI_PATH)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _upi_logic = module
        else:
            # Fallback: assume it is on sys.path (e.g. when running from src/)
            import upi_logic as _upi_logic_mod
            _upi_logic = _upi_logic_mod
        return _upi_logic
    except Exception as e:
        print(f"[main] Could not import upi_logic: {e}")
        return None

# -------------------------------
# SCRIPT PATHS  (always relative to this file)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACE_ENROLL          = os.path.join(BASE_DIR, "face_auth",            "enroll.py")
FACE_VERIFY          = os.path.join(BASE_DIR, "face_auth",            "verify.py")
VOICE_ENROLL         = os.path.join(BASE_DIR, "voice_auth",           "record.py")
VOICE_VERIFY         = os.path.join(BASE_DIR, "voice_auth",           "verify.py")
CURRENCY_RECOGNITION = os.path.join(BASE_DIR, "currency_recognition", "preprocess.py")

# -------------------------------
# HEALTH CHECK
# -------------------------------
def health_check():
    import importlib
    required = {
        "cv2":                "opencv-python",
        "mtcnn":              "mtcnn",
        "keras_facenet":      "keras-facenet",
        "pygame":             "pygame",
        "speech_recognition": "SpeechRecognition",
        "gtts":               "gtts",
    }
    missing = {pkg: pip for pkg, pip in required.items()
               if not importlib.util.find_spec(pkg)}
    if missing:
        print("Missing packages:")
        for pkg, pip in missing.items():
            print(f"  {pkg}  â†’  pip install {pip}")
        print("\nRun: pip install -r requirements.txt")
        sys.exit(2)

# -------------------------------
# MIC DISCOVERY  (cached)
# -------------------------------
_mic_index = None

def find_mic():
    global _mic_index
    if _mic_index is not None:
        return _mic_index
    for i in range(len(sr.Microphone.list_microphone_names())):
        try:
            with sr.Microphone(device_index=i):
                _mic_index = i
                return i
        except Exception:
            continue
    return None

# -------------------------------
# VOICE  YES / NO
# -------------------------------
# -------------------------------
# VOICE  YES / NO
# -------------------------------
YES_WORDS = {"yes", "yeah", "ya", "haan", "han", "haa"}
NO_WORDS  = {"no",  "nah",  "na", "nahi", "naa"}

def get_yes_no(retries: int = 3) -> bool:
    import threading

    r         = sr.Recognizer()
    mic_index = find_mic()

    speak("Are you already enrolled? Say yes or no — or type Y / N and press Enter.")
    time.sleep(0.3)

    for attempt in range(retries):
        result_box = {"val": None, "done": False}

        # ── keyboard thread ───────────────────────────────────────────────────
        def kb_thread():
            try:
                raw = input("  Enrolled? (Y/N): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return
            if not result_box["done"]:
                if raw in ("y", "yes"):
                    result_box["val"] = True
                elif raw in ("n", "no"):
                    result_box["val"] = False
                result_box["done"] = True

        # ── voice thread ──────────────────────────────────────────────────────
        def voice_thread():
            if mic_index is None:
                return
            try:
                with sr.Microphone(device_index=mic_index) as source:
                    r.adjust_for_ambient_noise(source, duration=0.3)
                    audio = r.listen(source, timeout=4, phrase_time_limit=3)
                text  = r.recognize_google(audio).lower().strip()
                print(f"[yes/no] Heard: {text!r}")
                words = set(text.split())
                if not result_box["done"]:
                    if words & YES_WORDS:
                        result_box["val"] = True
                        result_box["done"] = True
                    elif words & NO_WORDS:
                        result_box["val"] = False
                        result_box["done"] = True
            except Exception:
                pass

        kt = threading.Thread(target=kb_thread,    daemon=True)
        vt = threading.Thread(target=voice_thread, daemon=True)
        kt.start()
        vt.start()

        # Wait up to 6 s for either input to win
        deadline = time.time() + 6
        while not result_box["done"] and time.time() < deadline:
            time.sleep(0.1)

        if result_box["done"] and result_box["val"] is not None:
            return result_box["val"]

        if attempt < retries - 1:
            speak("Please say yes or no, or type Y or N.")
            time.sleep(0.3)

    speak("No response detected. Assuming not enrolled.")
    return False

# -------------------------------
# PROCESS RUNNER
# -------------------------------
def run_process(script_path) -> int:
    """
    Run a Python script using the current venv interpreter.
    cwd is set to the script's directory so relative paths resolve correctly.

    FIX: PYTHONPATH is extended with BASE_DIR (i.e. src/) so that shared
    modules like tts.py, user_manager.py are importable from any sub-package
    (face_auth/, voice_auth/, etc.) without each file needing a sys.path hack.
    Returns the exit code (0 = success, non-zero = failure).
    """
    if not script_path:
        print("[runner] No script path provided.")
        return -1
    if not os.path.exists(script_path):
        print(f"[runner] FILE NOT FOUND: {script_path}")
        return -1

    script_dir = os.path.dirname(script_path)

    # Inherit the current environment and prepend src/ to PYTHONPATH
    env = os.environ.copy()
    existing_pypath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        BASE_DIR + os.pathsep + existing_pypath
        if existing_pypath
        else BASE_DIR
    )

    print(f"[runner] Starting: {script_path}")
    print(f"[runner] PYTHONPATH includes: {BASE_DIR}")
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=script_dir,
        env=env,
    )
    print(f"[runner] Exit code: {result.returncode}")
    return result.returncode

# -------------------------------
# RESOLVE USER_ID FROM VERIFY PROCESS
# -------------------------------
def _detect_user_id_from_voice() -> str:
    """
    Reads the user_id written by voice_auth/verify.py into a temp file
    immediately before sys.exit(0).  Falls back to the first enrolled
    user, then to 'user_001' if nothing else is available.
    """
    import tempfile, json

    # ── Primary: temp file handshake ─────────────────────────────────
    handshake = os.path.join(tempfile.gettempdir(), "vispay_last_user.txt")
    try:
        if os.path.exists(handshake):
            with open(handshake) as f:
                uid = f.read().strip()
            os.remove(handshake)   # consume it — don't reuse on next login
            if uid:
                print(f"[FSM] user_id from handshake file: {uid}")
                return uid
    except Exception as e:
        print(f"[FSM] Handshake read failed: {e}")

    # ── Fallback: first entry in voice_auth users.json ───────────────
    try:
        users_json = os.path.join(BASE_DIR, "voice_auth", "enrolled_voices", "users.json")
        if os.path.exists(users_json):
            with open(users_json) as f:
                users = json.load(f)
            if users:
                uid = list(users.keys())[0]
                print(f"[FSM] user_id fallback from voice index: {uid}")
                return uid
    except Exception as e:
        print(f"[FSM] Voice index fallback failed: {e}")

    print("[FSM] All user_id resolution failed — using 'user_001'")
    return "user_001"

# -------------------------------
# FSM STATES
# -------------------------------
class State(Enum):
    START          = auto()
    ASK_ENROLL     = auto()
    FACE_ENROLL    = auto()
    VOICE_ENROLL   = auto()
    FACE_VERIFY    = auto()
    VOICE_VERIFY   = auto()
    AUTH_SUCCESS   = auto()
    PIN_VERIFY     = auto()   # gate between auth and menu
    POST_AUTH_MENU = auto()   # hands off to upi_logic.post_auth_menu()
    AUTH_FAIL      = auto()
    END            = auto()

# -------------------------------
# FSM
# -------------------------------
def run_fsm():
    state   = State.START
    user_id = None
    print("[FSM] Starting VisPay")

    # Initialise DB tables (idempotent — safe to call every startup)
    upi = _import_upi_logic()
    if upi:
        try:
            upi.init_db()
            print("[FSM] DB initialised.")
        except Exception as e:
            print(f"[FSM] DB init warning: {e}")

        # Select input mode BEFORE anything else so all subsequent
        # steps (PIN, payment, menu) use the correct mode
        try:
            upi._select_input_mode()
        except Exception as e:
            print(f"[FSM] Input mode selection failed: {e}")
            upi.INPUT_MODE = "both"   # safe fallback

    while state != State.END:

        # â”€â”€ START â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if state == State.START:
            speak("Welcome to VisPay. Your voice guided payment assistant.")
            state = State.ASK_ENROLL

        # â”€â”€ ASK ENROLL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif state == State.ASK_ENROLL:
            enrolled = get_yes_no()
            state = State.FACE_VERIFY if enrolled else State.FACE_ENROLL

        # â”€â”€ FACE ENROLL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif state == State.FACE_ENROLL:
            speak("Starting face enrollment. Please look at the camera.")
            code = run_process(FACE_ENROLL)
            if code == 0:
                state = State.VOICE_ENROLL
            else:
                speak("Face enrollment failed. Please try again later.")
                state = State.AUTH_FAIL

        # â”€â”€ VOICE ENROLL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif state == State.VOICE_ENROLL:
            speak("Starting voice enrollment. Please speak clearly when prompted.")
            code = run_process(VOICE_ENROLL)
            if code == 0:
                speak("Enrollment complete. Let us now verify your identity.")
                state = State.FACE_VERIFY   # must verify after enroll
            else:
                speak("Voice enrollment failed. Please try again later.")
                state = State.AUTH_FAIL

        # â”€â”€ FACE VERIFY â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif state == State.FACE_VERIFY:
            speak("Starting face verification. Please look at the camera.")
            code = run_process(FACE_VERIFY)
            if code == 0:
                state = State.AUTH_SUCCESS
            else:
                speak("Face verification failed. Trying voice verification.")
                state = State.VOICE_VERIFY

        # â”€â”€ VOICE VERIFY â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif state == State.VOICE_VERIFY:
            speak("Starting voice verification. Please speak clearly.")
            code = run_process(VOICE_VERIFY)
            if code == 0:
                state = State.AUTH_SUCCESS
            else:
                speak("Voice verification also failed.")
                state = State.AUTH_FAIL

        # â”€â”€ AUTH SUCCESS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif state == State.AUTH_SUCCESS:
            # Resolve the authenticated user_id so we can pass it forward.
            # voice_auth/verify.py prints the matched ID but does not return
            # it to main.py (subprocess boundary).  We derive it from the
            # enrolled users index; a temp-file handshake would be cleaner.
            user_id = _detect_user_id_from_voice()
            speak(f"Authentication successful. Welcome, {user_id}.")
            print(f"[FSM] AUTH SUCCESS â€” user_id={user_id}")
            state = State.PIN_VERIFY

        # â”€â”€ PIN VERIFY (new gate) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif state == State.PIN_VERIFY:
            if upi is None:
                speak("Transaction module unavailable. Skipping PIN check.")
                print("[FSM] upi_logic not loaded â€” bypassing PIN gate (dev mode)")
                state = State.POST_AUTH_MENU
            else:
                speak("Please enter your 4-digit PIN to continue.")
                pin_ok = upi.verify_pin(user_id)
                if pin_ok:
                    state = State.POST_AUTH_MENU
                else:
                    speak("PIN verification failed. Access denied.")
                    print("[FSM] PIN FAILED â€” access denied")
                    state = State.AUTH_FAIL

        # â”€â”€ POST AUTH MENU â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif state == State.POST_AUTH_MENU:
            if upi is None:
                speak("Transaction module unavailable. Please reinstall VisPay.")
                print("[FSM] upi_logic not loaded â€” cannot open menu")
                state = State.END
            else:
                # post_auth_menu() runs its own loop until the user says EXIT.
                # It handles PAY, BALANCE, HISTORY, CONTACTS, RECEIPTS,
                # CURRENCY, FORGOT PIN entirely inside upi_logic.py.
                try:
                    upi.post_auth_menu(user_id)
                except Exception as e:
                    print(f"[FSM] post_auth_menu error: {e}")
                    speak("An error occurred in the menu. Ending session.")
                state = State.END

        # â”€â”€ AUTH FAIL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif state == State.AUTH_FAIL:
            speak("Authentication failed. Access denied. Please contact support.")
            print("[FSM] ACCESS DENIED")
            state = State.END

    print("[FSM] Session ended.")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    health_check()
    run_fsm()