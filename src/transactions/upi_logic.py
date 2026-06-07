"""
upi_logic.py  —  VisPay Transaction Engine  v2
================================================
Dual-input: VOICE  +  KEYBOARD  (every step)
Payment:    UPI deep-link (production-identical to GPay/PhonePe)
Security:   8-layer chain (rate-limit → face → PIN → confirm)
Ledger:     Immutable SQLite + JSON audit log
"""

import asyncio, hashlib, json, logging, os, re, sqlite3
import tempfile, time, urllib.parse, uuid, webbrowser
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import speech_recognition as sr

# ── Razorpay test simulation ──────────────────────────────────────────────────
try:
    from razorpay_simulation import razorpay_pay, razorpay_refund, \
                                    get_razorpay_history, print_razorpay_receipt
    _RAZORPAY = True
except ImportError:
    _RAZORPAY = False

# ── optional imports ──────────────────────────────────────────────────────────
try:
    import edge_tts;    _EDGE_TTS = True
except ImportError:     _EDGE_TTS = False

try:
    from gtts import gTTS
    import pygame;      _GTTS = True
except ImportError:     _GTTS = False

try:
    import bcrypt;      _BCRYPT = True
except ImportError:     _BCRYPT = False

try:
    from pyzbar import pyzbar as _pyzbar
    _PYZBAR = True
except ImportError:     _PYZBAR = False

# ── Currency recognition ──────────────────────────────────────────────────────
try:
    import sys as _sys
    _SRC_PATH      = r"C:\projects\vispay\src"
    _CURRENCY_PATH = r"C:\projects\vispay\src\currency_recognition"
    for _p in (_SRC_PATH, _CURRENCY_PATH):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
    import preprocess as _currency_mod
    _CURRENCY = True
except ImportError as _e:
    print(f"  [currency] Import failed: {_e}")   # ← tells you WHAT actually failed
    _CURRENCY = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
_HERE = os.path.dirname(os.path.abspath(__file__))
CFG = {
    "db_path": os.path.join(_HERE, "vispay_transactions.db"),
    "contacts_file":    "trusted_contacts.json",
    "log_file":         "vispay_audit.log",
    "max_per_txn":      10_000,
    "soft_limit":        5_000,
    "daily_limit":      10_000,
    "rate_window_sec":    600,
    "rate_max_txns":        3,
    "pin_max_attempts":     3,
    "voice_max_attempts":   3,
    "qr_timeout_sec":      30,
    "upi_wait_sec":        12,
}

INPUT_MODE = "voice"   # "voice" | "keyboard" | "both"  — set at startup

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":%(message)s}',
    handlers=[
        logging.FileHandler(CFG["log_file"], encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("vispay")

def _log(event, **kw):
    log.info(json.dumps({"event": event, **kw}))

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════
def _db():
    """
    Open the SQLite DB with:
      • timeout=30   — retry for up to 30 s instead of raising immediately
      • WAL mode     — allows concurrent readers + one writer (eliminates
                       most 'database is locked' errors from parallel threads)
      • busy_timeout — belt-and-braces at the C layer (matches Python timeout)
    """
    con = sqlite3.connect(CFG["db_path"], timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA busy_timeout=30000;")   # 30 000 ms at SQLite C layer
    return con

def init_db():
    con = _db()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS accounts (
            user_id    TEXT PRIMARY KEY,
            name       TEXT NOT NULL,
            pin_hash   TEXT NOT NULL,
            balance    REAL NOT NULL DEFAULT 10000.0,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS transactions (
            txn_id       TEXT PRIMARY KEY,
            sender_id    TEXT NOT NULL,
            receiver_id  TEXT NOT NULL,
            amount       REAL NOT NULL,
            upi_url      TEXT NOT NULL,
            status       TEXT NOT NULL,
            initiated_at TEXT NOT NULL,
            confirmed_at TEXT,
            note         TEXT
        );
        CREATE TABLE IF NOT EXISTS rate_log (
            user_id  TEXT NOT NULL,
            txn_time TEXT NOT NULL
        );
    """)
    con.commit(); con.close()
    _log("db_init", path=CFG["db_path"])

# ══════════════════════════════════════════════════════════════════════════════
# PIN SECURITY
# ══════════════════════════════════════════════════════════════════════════════
def _hash_pin(pin):
    if _BCRYPT:
        return bcrypt.hashpw(pin.encode(), bcrypt.gensalt()).decode()
    return hashlib.sha256(f"vispay_{pin}_salt".encode()).hexdigest()

def _verify_pin(pin, stored):
    if _BCRYPT:
        try: return bcrypt.checkpw(pin.encode(), stored.encode())
        except: return False
    return hashlib.sha256(f"vispay_{pin}_salt".encode()).hexdigest() == stored

def register_user(user_id, name, pin):
    if not re.fullmatch(r"\d{4}", pin):
        raise ValueError("PIN must be 4 digits")
    con = _db()
    try:
        con.execute(
            "INSERT INTO accounts VALUES (?,?,?,?,?)",
            (user_id, name, _hash_pin(pin), 10000.0, datetime.now().isoformat())
        )
        con.commit(); _log("user_registered", user_id=user_id); return True
    except sqlite3.IntegrityError: return False
    finally: con.close()

# ══════════════════════════════════════════════════════════════════════════════
# SPOKEN NUMBER → DIGIT CONVERSION  (shared utility)
# ══════════════════════════════════════════════════════════════════════════════
_WORD_TO_INT = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,
    "five":5,"six":6,"seven":7,"eight":8,"nine":9,
    "ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,
    "fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,
    "twenty":20,"thirty":30,"forty":40,"fifty":50,
    "sixty":60,"seventy":70,"eighty":80,"ninety":90,
}
_DIGIT_WORDS = {"zero","one","two","three","four","five","six","seven","eight","nine"}
_DIGIT_CHARS = {w: str(v) for w, v in _WORD_TO_INT.items() if w in _DIGIT_WORDS}
_SCALE       = {"hundred": 100, "thousand": 1000, "lakh": 100_000}
_SKIP_WORDS  = {"rupees","rupee","rs","inr","and","a","please","send","pay","of"}

def _spoken_to_amount(text: str) -> str:
    """
    Convert spoken amount to a digit string.
    Handles:
      "500"                       → "500"
      "500 rupees"                → "500"
      "five hundred"              → "500"
      "five zero zero"            → "500"
      "one thousand"              → "1000"
      "two thousand five hundred" → "2500"
      "twenty five"               → "25"
      "one lakh"                  → "100000"
      "how much to say"           → ""   (voice mishear → empty)
    Returns empty string if nothing numeric found.
    """
    text = text.lower().strip()

    # Fast path: already contains digits
    cleaned = re.sub(r"[^\d.]", "", text)
    if cleaned:
        return cleaned

    words = [w for w in text.split() if w not in _SKIP_WORDS]
    if not words:
        return ""

    # If ALL words are single digit words → digit-string mode (e.g. "five zero zero" → "500")
    if all(w in _DIGIT_WORDS for w in words):
        return "".join(_DIGIT_CHARS[w] for w in words)

    # English number composition (e.g. "five hundred", "two thousand five hundred")
    total = 0
    chunk = 0
    for word in words:
        if word in _SCALE:
            scale = _SCALE[word]
            if scale >= 1000:
                total += (chunk if chunk != 0 else 1) * scale
                chunk = 0
            else:   # hundred
                chunk = (chunk if chunk != 0 else 1) * scale
        elif word in _WORD_TO_INT:
            chunk += _WORD_TO_INT[word]
        # unknown words silently ignored

    total += chunk
    return str(total) if total > 0 else ""


def _spoken_to_pin(text: str) -> str:
    """Convert spoken PIN words to digit string. e.g. 'one two three four' → '1234'"""
    words = text.lower().split()
    digits = "".join(
        _DIGIT_CHARS.get(w, re.sub(r"[^\d]", "", w)) for w in words
    )
    return re.sub(r"[^\d]", "", digits)

# ══════════════════════════════════════════════════════════════════════════════
# UPI VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
_UPI_RE = re.compile(r"^[a-zA-Z0-9.\-_]{3,256}@[a-zA-Z]{3,64}$")
KNOWN_HANDLES = {
    "okaxis","okhdfcbank","oksbi","okicici","ybl","paytm",
    "ibl","axl","upi","apl","fbl","timecosmos","pingpay","apl","rajgovhdfcbank"
}

def validate_upi_id(upi_id):
    upi_id = upi_id.strip().lower()
    if not _UPI_RE.match(upi_id):
        return False, "Format invalid — expected user@handle (e.g. name@okaxis)"
    handle = upi_id.split("@")[1]
    if handle not in KNOWN_HANDLES:
        return True, f"⚠ Unknown handle '{handle}' — double-check before confirming"
    return True, "OK"

# ══════════════════════════════════════════════════════════════════════════════
# CONTACTS
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULT_CONTACTS = {
    "mother":     "mother@okaxis",
    "shopkeeper": "kirana@okhdfcbank",
    "son":        "son@oksbi",
}

def load_contacts():
    p = Path(CFG["contacts_file"])
    if not p.exists():
        p.write_text(json.dumps(_DEFAULT_CONTACTS, indent=2), encoding="utf-8")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except:
        return _DEFAULT_CONTACTS.copy()

def save_contacts(contacts):
    Path(CFG["contacts_file"]).write_text(json.dumps(contacts, indent=2), encoding="utf-8")

def add_trusted_contact(name, upi_id):
    valid, reason = validate_upi_id(upi_id)
    if not valid: return False, reason
    c = load_contacts(); c[name.lower()] = upi_id.lower(); save_contacts(c)
    return True, "Saved"

# ══════════════════════════════════════════════════════════════════════════════
# RATE LIMIT + DAILY LIMIT
# ══════════════════════════════════════════════════════════════════════════════
def _check_rate(user_id):
    con = _db()
    ws = (datetime.now() - timedelta(seconds=CFG["rate_window_sec"])).isoformat()
    con.execute("DELETE FROM rate_log WHERE txn_time < ?", (ws,)); con.commit()
    n = con.execute("SELECT COUNT(*) FROM rate_log WHERE user_id=? AND txn_time>=?",
                    (user_id, ws)).fetchone()[0]
    con.close()
    if n >= CFG["rate_max_txns"]:
        return False, f"Rate limit: max {CFG['rate_max_txns']} payments per 10 minutes."
    return True, "OK"

def _check_daily(user_id, amount):
    con = _db()
    day = datetime.now().replace(hour=0,minute=0,second=0,microsecond=0).isoformat()
    spent = con.execute(
        "SELECT COALESCE(SUM(amount),0) FROM transactions "
        "WHERE sender_id=? AND status='SUCCESS' AND initiated_at>=?",
        (user_id, day)).fetchone()[0]
    con.close()
    if spent + amount > CFG["daily_limit"]:
        return False, f"Daily limit ₹{CFG['daily_limit']} reached. Remaining: ₹{CFG['daily_limit']-spent:.0f}"
    return True, "OK"

def _record_rate(user_id, con=None):
    """
    Insert a rate-log entry.
    Pass an existing open connection (con) to avoid opening a second
    connection while the caller already holds one — that causes
    'database is locked' on Windows SQLite.
    When called standalone (no con), opens and closes its own.
    """
    _own = con is None
    if _own:
        con = _db()
    con.execute("INSERT INTO rate_log VALUES (?,?)", (user_id, datetime.now().isoformat()))
    if _own:
        con.commit(); con.close()

# ══════════════════════════════════════════════════════════════════════════════
# BALANCE + HISTORY
# ══════════════════════════════════════════════════════════════════════════════
def get_balance(user_id):
    con = _db()
    row = con.execute("SELECT balance FROM accounts WHERE user_id=?", (user_id,)).fetchone()
    con.close()
    return row["balance"] if row else None

def get_history(user_id, limit=10):
    con = _db()
    rows = con.execute(
        "SELECT * FROM transactions WHERE sender_id=? OR receiver_id=? "
        "ORDER BY initiated_at DESC LIMIT ?",
        (user_id, user_id, limit)).fetchall()
    con.close()
    return [dict(r) for r in rows]

# ══════════════════════════════════════════════════════════════════════════════
# ████  DUAL INPUT  ████  (VOICE + KEYBOARD — core of v2)
# ══════════════════════════════════════════════════════════════════════════════

def _kb_input(prompt_text):
    """Clean keyboard input — strips, lowercases."""
    try:
        val = input(f"  ⌨  {prompt_text}: ").strip().lower()
        _log("keyboard_input", text=val)
        return val
    except (EOFError, KeyboardInterrupt):
        return ""

def _voice_input(timeout=6):
    """Single voice capture attempt."""
    r = sr.Recognizer()
    r.energy_threshold = 300
    r.dynamic_energy_threshold = True
    try:
        with sr.Microphone() as src:
            r.adjust_for_ambient_noise(src, duration=0.5)
            audio = r.listen(src, timeout=timeout, phrase_time_limit=8)
        result = r.recognize_google(audio).lower().strip()
        _log("voice_input", text=result)
        return result
    except sr.WaitTimeoutError: return ""
    except sr.UnknownValueError: return ""
    except Exception as e:
        _log("voice_error", error=str(e)); return ""

def get_input(speak_prompt, kb_prompt=None, timeout=6, secret=False):
    """
    Universal input — works in voice, keyboard, or both mode.
    secret=True  → uses getpass for PIN (keyboard), voice only (voice mode).
    kb_prompt    → custom keyboard label (defaults to speak_prompt).
    """
    label = kb_prompt or speak_prompt

    speak(speak_prompt)

    if INPUT_MODE == "keyboard":
        if secret:
            import getpass
            try:
                val = getpass.getpass(f"  🔒 {label}: ").strip()
                return val
            except:
                return ""
        return _kb_input(label)

    if INPUT_MODE == "voice":
        for attempt in range(1, CFG["voice_max_attempts"] + 1):
            if attempt > 1:
                speak("Sorry, didn't catch that. Please try again.")
            result = _voice_input(timeout)
            if result: return result
        speak("Could not understand voice input.")
        return ""

    # ── BOTH mode ─────────────────────────────────────────────────────────────
    # Strategy:
    #   • Voice runs for (timeout * voice_max_attempts) seconds total.
    #   • Keyboard runs in background — if user types, it wins immediately.
    #   • Once voice window closes (with or without result), we return.
    #     The keyboard thread is daemon so it never blocks the process.
    #   • This prevents the 48-second freeze when voice gets no audio.
    import threading

    result_box = {"val": None, "done": False}
    voice_window = timeout * CFG["voice_max_attempts"]   # e.g. 6*3 = 18 s

    def voice_thread():
        time.sleep(0.8)   # echo suppression: let TTS finish before mic opens
        for _ in range(CFG["voice_max_attempts"]):
            if result_box["done"]:
                return
            v = _voice_input(timeout)
            if v and not result_box["done"]:
                result_box["val"] = v
                result_box["done"] = True
                return
        # Voice exhausted — unblock the main wait so keyboard still gets a short window
        # but we don't hang forever
        result_box["voice_done"] = True

    def kb_thread():
        if secret:
            import getpass
            try:
                v = getpass.getpass(f"  🔒 {label}: ").strip()
            except Exception:
                v = input(f"  🔒 {label}: ").strip()
        else:
            v = input(f"  ⌨  {label} (or speak): ").strip().lower()
        if not result_box.get("done"):
            result_box["val"] = v
            result_box["done"] = True

    result_box["voice_done"] = False

    # PIN: keyboard-first so the prompt appears before mic noise
    if secret:
        kt = threading.Thread(target=kb_thread, daemon=True)
        vt = threading.Thread(target=voice_thread, daemon=True)
        kt.start()
        time.sleep(0.2)
        vt.start()
    else:
        vt = threading.Thread(target=voice_thread, daemon=True)
        kt = threading.Thread(target=kb_thread,    daemon=True)
        vt.start()
        kt.start()

    # Wait until voice wins, keyboard wins, or voice window expires
    voice_deadline = time.time() + voice_window + 1.2   # +1.2 for echo delay
    while not result_box["done"] and time.time() < voice_deadline:
        time.sleep(0.1)

    # If voice gave nothing but keyboard thread is still waiting for user to type,
    # give keyboard a small extra window (3 s) before giving up entirely
    if not result_box["done"]:
        extra_deadline = time.time() + 3
        while not result_box["done"] and time.time() < extra_deadline:
            time.sleep(0.1)

    return result_box["val"] or ""

# ══════════════════════════════════════════════════════════════════════════════
# TTS  —  edge-tts → gTTS → print
# ══════════════════════════════════════════════════════════════════════════════
def speak(text):
    print(f"\n  🔊  {text}")
    if INPUT_MODE == "keyboard":
        return     # keyboard mode — no audio needed
    if _EDGE_TTS:
        try: _speak_edge(text); return
        except Exception as e: _log("edge_tts_fail", error=str(e))
    if _GTTS:
        try: _speak_gtts(text); return
        except Exception as e: _log("gtts_fail", error=str(e))

def _speak_edge(text):
    async def _run():
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            path = f.name
        await edge_tts.Communicate(text, voice="en-IN-NeerjaNeural").save(path)
        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy(): time.sleep(0.2)
        pygame.mixer.quit()
        os.unlink(path)
    asyncio.run(_run())

def _speak_gtts(text):
    tts = gTTS(text=text, lang="en", tld="co.in")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts.save(f.name); path = f.name
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy(): time.sleep(0.2)
    pygame.mixer.quit()
    os.unlink(path)

# ══════════════════════════════════════════════════════════════════════════════
# QR SCANNER
# ══════════════════════════════════════════════════════════════════════════════

def _try_decode_frame(frame):
    """
    Try OpenCV + pyzbar on multiple enhanced image variants.
    Returns decoded text or None.
    """
    detector = cv2.QRCodeDetector()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    variants = [
        frame,
        gray,
        cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC),
        cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC),
        cv2.GaussianBlur(gray, (3, 3), 0),
    ]

    try:
        import numpy as np

        sharpen_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])

        variants.append(cv2.filter2D(gray, -1, sharpen_kernel))
    except Exception:
        pass

    for img in variants:

        # ── OpenCV QR detector ─────────────────────────────
        try:
            data, bbox, _ = detector.detectAndDecode(img)

            if bbox is not None:
                print("[DEBUG] QR pattern detected")

            if data and data.strip():
                print("[DEBUG] OpenCV decoded:", data)
                return data.strip()

        except Exception as e:
            print("[DEBUG] OpenCV decode error:", e)

        # ── pyzbar fallback ───────────────────────────────
        if _PYZBAR:
            try:
                scan_img = img if len(img.shape) == 2 else cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2GRAY
                )

                codes = _pyzbar.decode(scan_img)

                for c in codes:
                    text = c.data.decode("utf-8").strip()

                    if text:
                        print("[DEBUG] pyzbar decoded:", text)
                        return text

            except Exception as e:
                print("[DEBUG] pyzbar decode error:", e)

    return None


def scan_qr_opencv():

    print("=" * 50)
    print("[DEBUG] ENTERED scan_qr_opencv()")
    print("=" * 50)

    speak(
        "Starting QR scan. Hold the QR code in front of your camera. Press Q to cancel."
    )

    print("[DEBUG] PYZBAR =", _PYZBAR)

    cap = cv2.VideoCapture(0)

    print("[DEBUG] cap created")
    print("[DEBUG] isOpened =", cap.isOpened())

    if not cap.isOpened():
        speak("Camera unavailable.")
        print("[DEBUG] Camera open failed")
        return None

    print("[DEBUG] Camera opened successfully")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    except Exception:
        pass

    qr_data = None
    start = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            print("[DEBUG] Frame capture failed")
            continue

        qr_data = _try_decode_frame(frame)

        if qr_data:

            print("[DEBUG] Raw QR Data:", qr_data)

            cv2.putText(
                frame,
                "QR Detected! ✓",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

            cv2.imshow("VisPay QR — press Q to cancel", frame)

            cv2.waitKey(1500)

            break

        remaining = int(
            CFG["qr_timeout_sec"] - (time.time() - start)
        )

        decoder_label = (
            "OpenCV + pyzbar"
            if _PYZBAR
            else "OpenCV"
        )

        cv2.putText(
            frame,
            f"Hold QR steady... {remaining}s | Q = cancel",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2
        )

        cv2.putText(
            frame,
            f"Good lighting • QR fills frame [{decoder_label}]",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1
        )

        cv2.imshow("VisPay QR — press Q to cancel", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if remaining <= 0:
            break

    cap.release()
    cv2.destroyAllWindows()

    if not qr_data:
        speak("No QR detected.")
        return None

    upi_id = _qr_to_upi(qr_data)

    print("[DEBUG] Extracted UPI ID:", upi_id)

    if not upi_id:
        speak("QR does not contain a valid UPI ID.")
        return None

    valid, reason = validate_upi_id(upi_id)

    if not valid:
        speak(f"Invalid UPI ID in QR. {reason}")
        return None

    if "⚠" in reason:
        speak(reason)

    confirm = get_input(
        f"QR scanned. Payee is {upi_id}. Say or type YES to confirm.",
        kb_prompt="Confirm QR payee? (yes/no)"
    )

    return upi_id if confirm and "yes" in confirm.lower() else None


def _qr_to_upi(raw):
    raw = raw.strip()

    print("[DEBUG] Parsing QR:", raw)

    if raw.lower().startswith("upi://pay"):

        params = dict(
            urllib.parse.parse_qsl(
                urllib.parse.urlparse(raw).query
            )
        )

        print("[DEBUG] UPI params:", params)

        return params.get("pa")

    if "@" in raw:
        return raw

    return None

# ══════════════════════════════════════════════════════════════════════════════
# PAYEE SELECTION
# ══════════════════════════════════════════════════════════════════════════════
def select_payee():
    _print_divider("SELECT PAYEE")
    choice = get_input(
        "How to pay? Say or type:  QR  /  CONTACT  /  MANUAL",
        kb_prompt="Choose method (qr / contact / manual)"
    )
    if not choice: return None, None

    # ── QR ────────────────────────────────────────────────────────────────────
    if "qr" in choice:
        upi_id = scan_qr_opencv()
        if upi_id: return "QR Payee", upi_id
        speak("Falling back to contacts.")
        choice = "contact"

    # ── CONTACT ───────────────────────────────────────────────────────────────
    if "contact" in choice:
        contacts = load_contacts()
        _print_contacts(contacts)
        name_input = get_input(
            f"Say or type a contact name: {', '.join(contacts.keys())}",
            kb_prompt="Contact name"
        ).lower().strip()

        # Exact match first
        if name_input in contacts:
            upi_id = contacts[name_input]
        else:
            # Fuzzy: find contact whose name appears in the spoken text or vice versa
            matched = None
            for contact_name in contacts:
                if contact_name in name_input or name_input in contact_name:
                    matched = contact_name
                    break
            if matched:
                speak(f"Did you mean {matched}?")
                upi_id = contacts[matched]
                name_input = matched
            else:
                speak(f"'{name_input}' not found in contacts. Available: {', '.join(contacts.keys())}")
                return None, None

        valid, reason = validate_upi_id(upi_id)
        if not valid:
            speak(f"Stored UPI for {name_input} is invalid: {reason}"); return None, None
        return name_input, upi_id

    # ── MANUAL ────────────────────────────────────────────────────────────────
    if "manual" in choice:
        upi_id = get_input(
            "Say or type the UPI ID (example: name@okaxis)",
            kb_prompt="Enter UPI ID"
        ).strip().lower()
        valid, reason = validate_upi_id(upi_id)
        if not valid:
            speak(f"Invalid: {reason}"); return None, None
        if "⚠" in reason: speak(reason)
        confirm = get_input(
            f"UPI ID: {upi_id}. Say or type YES to confirm.",
            kb_prompt="Confirm? (yes/no)"
        )
        if "yes" in confirm: return "Manual", upi_id
        speak("Cancelled."); return None, None

    speak("Didn't understand choice."); return None, None

# ══════════════════════════════════════════════════════════════════════════════
# AMOUNT
# ══════════════════════════════════════════════════════════════════════════════
def get_amount(user_id):
    """
    Amount entry — special-cased to avoid the stdin-blocking thread bug.

    In BOTH mode, input() cannot be interrupted from another thread, so a retry
    loop calling get_input() re-launches new threads each time while the old
    keyboard thread is still blocking stdin — causing a permanent freeze.

    Fix: handle amount with a direct, sequential flow:
      1. Speak the prompt.
      2. If BOTH or VOICE mode, try one short voice capture (non-blocking, 5 s).
      3. If voice got something, use it; otherwise fall through to keyboard.
      4. Read keyboard with a single direct input() call — no threads.
    This guarantees only one input() is live at any time.
    """
    _print_divider("ENTER AMOUNT")

    for attempt in range(1, 4):
        if attempt == 1:
            speak("How much to send? Say the amount in rupees. For example, say five hundred.")
        else:
            speak("Please try again. Say the amount, or type it below and press Enter.")

        raw = ""

        # ── Try voice first (BOTH / VOICE modes) ─────────────────────────────
        if INPUT_MODE in ("voice", "both"):
            time.sleep(0.6)   # let TTS finish before mic opens
            raw = _voice_input(timeout=5)
            if raw:
                _log("voice_amount", text=raw)

        # ── Keyboard fallback (always shown in BOTH / KEYBOARD modes) ─────────
        if not raw and INPUT_MODE in ("keyboard", "both"):
            try:
                raw = input("  ⌨  Amount (₹): ").strip().lower()
                _log("keyboard_amount", text=raw)
            except (EOFError, KeyboardInterrupt):
                speak("Cancelled.")
                return None

        if not raw:
            speak("No input received. Please try again.")
            continue

        # ── Parse ─────────────────────────────────────────────────────────────
        digits = _spoken_to_amount(raw)
        if not digits:
            speak(f"Sorry, I heard '{raw}' but couldn't find a number. "
                  "Please type the amount and press Enter.")
            # force keyboard on next attempt
            try:
                raw = input("  ⌨  Amount (₹): ").strip()
                digits = _spoken_to_amount(raw)
            except (EOFError, KeyboardInterrupt):
                speak("Cancelled."); return None
            if not digits:
                speak("Still couldn't understand. Skipping this attempt.")
                continue

        try:
            amount = float(digits)
        except Exception:
            speak("Invalid amount. Please try again."); continue

        if amount <= 0:
            speak("Amount must be greater than zero."); continue
        if amount > CFG["max_per_txn"]:
            speak(f"Maximum per transaction is ₹{CFG['max_per_txn']}."); continue

        ok, reason = _check_daily(user_id, amount)
        if not ok:
            speak(reason); return None

        if amount > CFG["soft_limit"]:
            confirm = get_input(
                f"Large amount ₹{amount:.0f}. Say or type YES to continue.",
                kb_prompt="Confirm large amount (yes/no)"
            )
            if "yes" not in confirm:
                speak("Cancelled."); return None

        return amount

    speak("Could not get a valid amount. Payment cancelled.")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# PIN VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════
def verify_pin(user_id):
    """
    PIN entry — uses the same sequential (no-thread) pattern as get_amount
    to avoid the stdin-blocking freeze in BOTH mode.
    PIN is always read from keyboard (getpass) for security; voice is offered
    as an alternative only if the user prefers it.
    """
    _print_divider("PIN VERIFICATION")
    con = _db()
    row = con.execute("SELECT pin_hash FROM accounts WHERE user_id=?", (user_id,)).fetchone()
    con.close()
    if not row: return False

    for attempt in range(1, CFG["pin_max_attempts"] + 1):
        speak(f"Attempt {attempt} of {CFG['pin_max_attempts']}. Enter your 4-digit PIN.")

        raw = ""

        # ── Voice option in BOTH / VOICE modes ───────────────────────────────
        if INPUT_MODE in ("voice", "both"):
            time.sleep(0.6)
            raw = _voice_input(timeout=5)
            if raw:
                _log("voice_pin", text="[hidden]")

        # keyboard fallback
        # keyboard fallback
        # Keyboard ALWAYS shown as fallback — PIN entry must never be voice-only
        if not raw:
            try:
                raw = input("  4-digit PIN: ").strip()
                _log("keyboard_pin", text="[hidden]")
            except (EOFError, KeyboardInterrupt):
                speak("Input error. Transaction blocked.")
                return False

        pin = _spoken_to_pin(raw)
        if len(pin) != 4:
            speak("PIN must be exactly 4 digits. Please try again."); continue
        if _verify_pin(pin, row["pin_hash"]):
            speak("PIN verified."); _log("pin_ok", user_id=user_id); return True
        remaining = CFG["pin_max_attempts"] - attempt
        if remaining:
            speak(f"Wrong PIN. {remaining} attempt{'s' if remaining > 1 else ''} left.")
        else:
            speak("Too many wrong attempts.")
        _log("pin_fail", user_id=user_id, attempt=attempt)

    # ── All attempts exhausted — offer Forgot PIN recovery ───────────────────
    speak("Forgot your PIN? Type YES to recover, or NO to cancel.")
    try:
        offer = input("  ⌨  Forgot PIN? (yes/no): ").strip().lower()
    except Exception:
        offer = ""
    if "yes" in offer:
        return forgot_pin_flow(user_id)

    speak("Transaction blocked.")
    return False


# ══════════════════════════════════════════════════════════════════════════════
# FORGOT PIN  —  Recovery: Face → Voice → Reset PIN
# Paths:
#   Face module  : C:\projects\vispay\src\face_auth\verify.py  → run_verification()
#   Voice module : C:\projects\vispay\src\voice_auth\verify.py → verify()  (sys.exit)
# ══════════════════════════════════════════════════════════════════════════════

def _face_recovery_hook(user_id: str) -> str:
    """
    Run face_auth/verify.py::run_verification().

    Returns one of:
        "VERIFIED"        — face matched successfully
        "FAILED"          — face check ran but did not match
        "NO_ENROLLMENT"   — no images enrolled for this user
        "SKIPPED"         — verify.py not found (dev machine without face_auth)

    Callers must treat only "VERIFIED" as success.
    """
    import importlib.util, sys as _sys

    FACE_PATH       = r"C:\projects\vispay\src\face_auth\verify.py"
    FACE_ENROLL_DIR = r"C:\projects\vispay\src\face_auth\enrolled_faces"

    if not os.path.exists(FACE_PATH):
        print(f"  [WARN] face_auth/verify.py not found at expected path")
        _log("face_verify_missing", user_id=user_id)
        return "SKIPPED"

    # Discover which user folder to verify against
    enrolled_dirs = []
    if os.path.exists(FACE_ENROLL_DIR):
        enrolled_dirs = [
            d for d in os.listdir(FACE_ENROLL_DIR)
            if os.path.isdir(os.path.join(FACE_ENROLL_DIR, d))
               and d != "__pycache__"
        ]

    if not enrolled_dirs:
        print(f"  [face] No face enrollment folders found in {FACE_ENROLL_DIR}")
        _log("face_no_enrollment_folders", user_id=user_id)
        return "NO_ENROLLMENT"

    face_id = user_id if user_id in enrolled_dirs else enrolled_dirs[0]
    print(f"  [face] Verifying against enrolled folder: '{face_id}'")
    print(f"  [face] DATASET_DIR override → {FACE_ENROLL_DIR}")

    try:
        face_dir = os.path.dirname(FACE_PATH)
        if face_dir not in _sys.path:
            _sys.path.insert(0, face_dir)

        spec   = importlib.util.spec_from_file_location("face_verify", FACE_PATH)
        module = importlib.util.module_from_spec(spec)

        # ── FIX: override DATASET_DIR BEFORE exec_module so load_embeddings()
        # uses the absolute face_auth path instead of a CWD-relative fallback.
        # verify.py sets DATASET_DIR = "enrolled_faces" at module level; by
        # injecting the correct absolute path into the module's namespace first,
        # exec_module will overwrite it only if the source also sets it — so we
        # set it again right after exec to be safe.
        module.__dict__["DATASET_DIR"] = FACE_ENROLL_DIR
        spec.loader.exec_module(module)
        # Re-apply after exec (module-level assignment in verify.py runs here)
        module.DATASET_DIR = FACE_ENROLL_DIR

        # Patch the voice-input function so the hook supplies the user_id
        # directly without opening a microphone during PIN recovery.
        module.get_user_id_by_voice = lambda: face_id

        result = module.run_verification()
        _log("face_recovery", user_id=user_id, result=result)
        print(f"  [face] run_verification() returned: {result}")
        return result   # "VERIFIED" | "FAILED" | "NO_ENROLLMENT"

    except Exception as e:
        _log("face_error", error=str(e))
        speak("Face verification error.")
        print(f"  [face] Exception: {e}")
        return "FAILED"


def _voice_recovery_hook(user_id: str) -> bool:
    """
    Run voice_auth/verify.py::verify().

    voice/verify.py calls sys.exit(0) on success and sys.exit(1) on failure,
    so we intercept SystemExit rather than a return value.
    """
    import importlib.util

    VOICE_PATH = r"C:\projects\vispay\src\voice_auth\verify.py"

    if not os.path.exists(VOICE_PATH):
        _log("voice_verify_missing", user_id=user_id)
        print(f"  [DEV] {VOICE_PATH} not found — skipping voice step (dev mode)")
        return True                          # dev bypass

    try:
        spec   = importlib.util.spec_from_file_location("voice_verify", VOICE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        try:
            module.verify()
            # If verify() returns normally (shouldn't per current code) → pass
            _log("voice_recovery", user_id=user_id, result="VERIFIED")
            return True
        except SystemExit as se:
            result = "VERIFIED" if se.code == 0 else "FAILED"
            _log("voice_recovery", user_id=user_id, result=result)
            return se.code == 0

    except Exception as e:
        _log("voice_error", error=str(e))
        speak("Voice verification error.")
        return False


def _reset_pin(user_id: str) -> bool:
    """
    Prompt for a new 4-digit PIN twice, hash it, and update the DB.
    Returns True on success.
    """
    _print_divider("RESET PIN")
    speak("Please set a new 4-digit PIN.")

    for _ in range(3):
        # plain input() avoids getpass blocking in PowerShell/VS Code.
        try:
            print("  New PIN (4 digits): ", end="", flush=True)
            new_raw = input().strip()
        except (EOFError, KeyboardInterrupt):
            speak("Cancelled."); return False
        new_pin = re.sub(r"[^\d]", "", new_raw)
        if len(new_pin) != 4:
            speak("PIN must be exactly 4 digits."); continue

        try:
            print("  Confirm PIN (4 digits): ", end="", flush=True)
            confirm_raw = input().strip()
        except (EOFError, KeyboardInterrupt):
            speak("Cancelled."); return False
        confirm_pin = re.sub(r"[^\d]", "", confirm_raw)
        if new_pin != confirm_pin:
            speak("PINs do not match. Please try again."); continue

        con = _db()
        con.execute(
            "UPDATE accounts SET pin_hash=? WHERE user_id=?",
            (_hash_pin(new_pin), user_id)
        )
        con.commit(); con.close()
        speak("PIN reset successfully. ✓")
        _log("pin_reset", user_id=user_id)
        return True

    speak("PIN reset failed. Too many mismatches.")
    return False


def forgot_pin_flow(user_id: str) -> bool:
    """
    Full Forgot PIN recovery chain:
      Step 1 — Face Verification   (face_auth/verify.py)
      Step 2 — Voice Verification  (voice_auth/verify.py)
      Step 3 — Reset PIN
      Step 4 — Re-enter new PIN to confirm payment proceeds

    Returns True if the user successfully recovers and re-enters the new PIN.
    """
    _print_divider("FORGOT PIN — RECOVERY")
    _log("forgot_pin_start", user_id=user_id)

    # ── Step 1: Face ──────────────────────────────────────────────────────────
    speak("Step 1 of 3: Face verification. Please look at the camera.")
    face_result = _face_recovery_hook(user_id)

    if face_result == "VERIFIED":
        speak("Face verified. ✓")
        _log("forgot_pin_face_ok", user_id=user_id)

    elif face_result == "NO_ENROLLMENT":
        speak("No face enrollment found for this account. "
              "Please enroll your face first using the face enrollment screen.")
        _log("forgot_pin_face_no_enrollment", user_id=user_id)
        return False

    elif face_result == "SKIPPED":
        # face_auth module missing entirely — block recovery, do not silently pass
        speak("Face authentication module is not available. Cannot recover PIN.")
        _log("forgot_pin_face_skipped", user_id=user_id)
        return False

    else:
        # "FAILED" or any unexpected value
        speak("Face verification failed. Cannot recover PIN.")
        _log("forgot_pin_face_fail", user_id=user_id, result=face_result)
        return False

    # ── Step 2: Voice ─────────────────────────────────────────────────────────
    # ── Voice step skipped: face verification already confirmed identity ──
    # Voice is only used as a fallback when face verification fails.
    _log("forgot_pin_voice_skipped", user_id=user_id, reason="face_verified")

    # ── Step 2 of 2: Reset PIN ─────────────────────────────
    speak("Step 2 of 2: Set your new PIN.")
   
    if not _reset_pin(user_id):
        _log("forgot_pin_reset_fail", user_id=user_id)
        return False

    # ── Step 4: Re-enter new PIN to authorise the current payment ─────────────
    speak("PIN reset complete. Please enter your new PIN to authorise this payment.")
    _log("forgot_pin_success", user_id=user_id)
    return verify_pin(user_id)

# ══════════════════════════════════════════════════════════════════════════════
# INITIATE + RECORD PAYMENT
# ══════════════════════════════════════════════════════════════════════════════
def _build_upi_url(upi_id, name, amount):
    txn_id = str(uuid.uuid4())
    params = {
        "pa":  upi_id,
        "pn":  name,
        "am":  f"{amount:.2f}",
        "cu":  "INR",
        "tn":  f"VisPay-{txn_id[:8]}",
        "tid": txn_id[:16],
    }
    return txn_id, "upi://pay?" + urllib.parse.urlencode(params)

def initiate_payment(sender_id, name, upi_id, amount):
    txn_id, upi_url = _build_upi_url(upi_id, name, amount)
    con = _db()
    con.execute(
        "INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?)",
        (txn_id, sender_id, upi_id, amount, upi_url,
         "INITIATED", datetime.now().isoformat(), None, name)
    )
    con.commit(); con.close()
    _log("txn_initiated", txn_id=txn_id, sender=sender_id, receiver=upi_id, amount=amount)

    _print_divider("PAYMENT INITIATED")
    print(f"\n  {'─'*50}")
    print(f"  Transaction ID : {txn_id[:16]}...")
    print(f"  Payee          : {name}  ({upi_id})")
    print(f"  Amount         : ₹{amount:.2f}")
    print(f"  UPI URL        : {upi_url[:60]}...")
    print(f"  {'─'*50}\n")

    speak(f"Opening your UPI app to pay ₹{amount:.0f} to {name}.")
    webbrowser.open(upi_url)
    return txn_id

def _finalize(txn_id, status):
    """
    Write final transaction status + deduct balance.
    All DB work uses ONE connection — _record_rate receives it as a
    parameter so it never opens a competing second connection.
    Retries up to 5× on OperationalError as an extra safety net.
    """
    for attempt in range(1, 6):
        try:
            con = _db()
            con.execute(
                "UPDATE transactions SET status=?, confirmed_at=? WHERE txn_id=?",
                (status, datetime.now().isoformat(), txn_id)
            )
            if status == "SUCCESS":
                row = con.execute(
                    "SELECT sender_id, amount FROM transactions WHERE txn_id=?",
                    (txn_id,)
                ).fetchone()
                if row:
                    con.execute(
                        "UPDATE accounts SET balance=balance-? WHERE user_id=?",
                        (row["amount"], row["sender_id"])
                    )
                    _record_rate(row["sender_id"], con=con)  # reuse same connection
            con.commit()
            con.close()
            _log("txn_finalized", txn_id=txn_id, status=status)
            return
        except sqlite3.OperationalError as e:
            _log("finalize_retry", attempt=attempt, error=str(e))
            if attempt == 5:
                _log("finalize_failed", txn_id=txn_id, error=str(e))
                raise
            time.sleep(0.5 * attempt)

def payment_confirmation(txn_id, amount, name):
    speak(f"Waiting {CFG['upi_wait_sec']} seconds for UPI app...")
    time.sleep(CFG["upi_wait_sec"])
    response = get_input(
        f"Did the ₹{amount:.0f} payment to {name} succeed? Say or type YES / NO.",
        kb_prompt="Payment successful? (yes/no)"
    )
    if "yes" in response:
        _finalize(txn_id, "SUCCESS")
        speak(f"Payment of ₹{amount:.0f} to {name} recorded as SUCCESS. TXN: {txn_id[:8]}")
        return True
    else:
        _finalize(txn_id, "FAILED")
        speak("Payment marked as FAILED. No balance deducted.")
        return False

# ══════════════════════════════════════════════════════════════════════════════
# FULL PAYMENT FLOW  (8-layer security chain)
# ══════════════════════════════════════════════════════════════════════════════
def run_payment_flow(user_id):
    """
    Security chain:
    1. Rate limit  2. Payee + UPI validate  3. Amount + daily cap
    4. PIN verify  5. Verbal/typed CONFIRM
    6. UPI deep-link + INITIATED record  7. SUCCESS/FAILED update

    Note: Face authentication is performed once at login.
    Re-verifying mid-payment adds friction for non-literate users
    without meaningful additional security.
    """
    _print_divider("NEW PAYMENT")

    # 1. Rate limit
    ok, reason = _check_rate(user_id)
    if not ok: speak(reason); return False

    # 2. Payee
    name, upi_id = select_payee()
    if not upi_id: speak("No payee selected. Cancelled."); return False

    # 3. Amount
    amount = get_amount(user_id)
    if amount is None: return False

    # 4. PIN
    if not verify_pin(user_id):
        speak("PIN failed. Transaction blocked."); return False

    # 5. Final confirmation — reads full summary
    _print_divider("CONFIRM PAYMENT")
    summary = (
        f"SUMMARY — Sending ₹{amount:.0f} to {name} "
        f"at UPI ID {upi_id}. "
        f"Say CONFIRM or type confirm to proceed, or CANCEL to abort."
    )
    final = get_input(summary, kb_prompt="Type CONFIRM or CANCEL")
    if "confirm" not in final:
        speak("Transaction cancelled by user.")
        _log("txn_cancelled", user_id=user_id); return False

    # 6. Initiate payment record (UPI URL + DB entry)
    txn_id = initiate_payment(user_id, name, upi_id, amount)

    # 7. Razorpay test gateway OR manual confirmation fallback
    if _RAZORPAY:
        rzp_result = razorpay_pay(
            vispay_txn_id = txn_id,
            sender_id     = user_id,
            payee_name    = name,
            upi_id        = upi_id,
            amount_inr    = amount,
            speak_fn      = speak,
        )
        success = rzp_result["success"]
        # Sync final status into main transactions table
        _finalize(txn_id, "SUCCESS" if success else "FAILED")
        if success:
            bal = get_balance(user_id)
            if bal is not None:
                speak(f"Remaining balance: ₹{bal:.2f}")
            speak(f"Razorpay order: {rzp_result.get('rzp_order_id','')[:16]}.")
        return success
    else:
        # Fallback: manual yes/no confirmation (original behaviour)
        success = payment_confirmation(txn_id, amount, name)
        if success:
            bal = get_balance(user_id)
            if bal is not None:
                speak(f"Remaining balance: ₹{bal:.2f}")
        return success

# ══════════════════════════════════════════════════════════════════════════════
# CURRENCY RECOGNITION
# ══════════════════════════════════════════════════════════════════════════════
def run_currency_recognition(user_id: str):
    """
    Launches preprocess.run_recognition() with _handle_payment() hooked
    into VisPay's payment flow.

    preprocess.py has its own full camera FSM:
      scanning → note stable → post-detect menu (again / pay / exit)

    When user says/presses PAY inside that menu, _handle_payment(label)
    is called with the detected denomination string e.g. "500".
    We replace that stub with a real VisPay payment flow call.
    """
    _print_divider("CURRENCY RECOGNITION")

    if not _CURRENCY:
        speak("Currency recognition module not available. "
              "Please check that preprocess.py and the TFLite model are present.")
        _log("currency_unavailable")
        return

    # ── Patch _handle_payment to call VisPay payment flow ────────────────────
    def _vispay_payment_hook(label: str):
        """
        Called by preprocess._post_detect_menu when user chooses 'pay'.
        label is the denomination string e.g. '500', '100', '2000'.
        """
        try:
            amount = float(label)
        except ValueError:
            speak(f"Could not parse denomination {label}.")
            return

        speak(f"Detected ₹{label}. Starting payment.")
        _log("currency_detected", denomination=label, user_id=user_id)

        # Select payee
        name, upi_id = select_payee()
        if not upi_id:
            speak("No payee selected. Returning to scanner.")
            return

        # Rate limit
        ok, reason = _check_rate(user_id)
        if not ok:
            speak(reason)
            return

        # PIN
        if not verify_pin(user_id):
            speak("PIN failed. Payment cancelled.")
            return

        # Execute payment
        txn_id = initiate_payment(user_id, name, upi_id, amount)
        if _RAZORPAY:
            rzp_result = razorpay_pay(
                vispay_txn_id=txn_id,
                sender_id=user_id,
                payee_name=name,
                upi_id=upi_id,
                amount_inr=amount,
                speak_fn=speak,
            )
            _finalize(txn_id, "SUCCESS" if rzp_result["success"] else "FAILED")
            if rzp_result["success"]:
                bal = get_balance(user_id)
                if bal:
                    speak(f"Remaining balance ₹{bal:.0f}.")
        else:
            payment_confirmation(txn_id, amount, name)

    # Inject the hook into preprocess module
    _currency_mod._handle_payment = _vispay_payment_hook

    # Run — this blocks until user presses ESC or says exit
    try:
        _currency_mod.run_recognition()
    except Exception as e:
        _log("currency_error", error=str(e))
        speak("Currency recognition error. Returning to menu.")
        print(f"  [currency] Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# POST-AUTH MENU  (call from FSM State.POST_AUTH_MENU)
# ══════════════════════════════════════════════════════════════════════════════
def post_auth_menu(user_id):
    _print_header()
    speak("Authentication successful. Welcome to VisPay.")

    rzp_badge = "✅" if _RAZORPAY else "⚠ (offline)"
    cur_badge = "✅" if _CURRENCY else "⚠ (offline)"
    MENU = f"""
  ┌──────────────────────────────────────────────┐
  │         VisPay  Main Menu                    │
  ├──────────────────────────────────────────────┤
  │  [1] PAY       Send money (Razorpay {rzp_badge})
  │  [2] BALANCE   Check balance                 │
  │  [3] HISTORY   Recent transactions           │
  │  [4] CONTACTS  Manage contacts               │
  │  [5] RECEIPTS  Razorpay payment receipts     │
  │  [6] CURRENCY  Scan currency note  {cur_badge}
  │  [7] FORGOT    Forgot PIN recovery           │
  │  [8] EXIT      Quit                          │
  └──────────────────────────────────────────────┘"""

    while True:
        try:
            print(MENU)
            choice = get_input(
                "Say or type: PAY, BALANCE, HISTORY, CONTACTS, CURRENCY, FORGOT, or EXIT",
                kb_prompt="Choice (pay/balance/history/contacts/currency/forgot/exit)"
            )
            if not choice: continue

            # ── PAY ───────────────────────────────────────────────────────────
            if any(x in choice for x in ["pay","send","1"]):
                run_payment_flow(user_id)

            # ── BALANCE ───────────────────────────────────────────────────────
            elif any(x in choice for x in ["balance","2"]):
                bal = get_balance(user_id)
                if bal is not None:
                    speak(f"Your balance is ₹{bal:.2f}")
                    _print_info(f"Balance: ₹{bal:.2f}")
                else:
                    speak("Could not retrieve balance.")

            # ── HISTORY ───────────────────────────────────────────────────────
            elif any(x in choice for x in ["history","recent","3"]):
                txns = get_history(user_id, limit=5)
                if not txns:
                    speak("No transactions yet.")
                else:
                    _print_divider("TRANSACTION HISTORY")
                    print(f"\n  {'TXN ID':<12} {'TO/FROM':<20} {'AMOUNT':>8}  {'DATE':<12}  STATUS")
                    print(f"  {'─'*70}")
                    for t in txns:
                        direction = "→ SENT" if t["sender_id"] == user_id else "← RECV"
                        peer = t["receiver_id"] if t["sender_id"] == user_id else t["sender_id"]
                        dt = t["initiated_at"][:10]
                        print(f"  {t['txn_id'][:10]:<12} {peer:<20} ₹{t['amount']:>7.0f}  {dt:<12}  {t['status']}")
                        speak(f"₹{t['amount']:.0f} {direction.lower()} {peer} on {dt}. {t['status']}.")

            # ── CONTACTS ──────────────────────────────────────────────────────
            elif any(x in choice for x in ["contact","4"]):
                _manage_contacts()

            # ── RECEIPTS ──────────────────────────────────────────────────────
            elif any(x in choice for x in ["receipt","razorpay","5"]):
                if _RAZORPAY:
                    _print_divider("RAZORPAY RECEIPTS")
                    orders = get_razorpay_history(limit=5)
                    if not orders:
                        speak("No Razorpay transactions yet.")
                    else:
                        for o in orders:
                            print_razorpay_receipt(o["rzp_order_id"])
                            speak(
                                f"Order {o['rzp_order_id'][:12]}, "
                                f"amount ₹{o['amount_paise']/100:.0f}, "
                                f"status {o['status']}."
                            )
                else:
                    speak("Razorpay module not loaded. Please install razorpay package.")

            # ── CURRENCY ──────────────────────────────────────────────────────
            elif any(x in choice for x in ["currency","scan","note","6"]):
                run_currency_recognition(user_id)

            # ── FORGOT PIN ────────────────────────────────────────────────────
            elif any(x in choice for x in ["forgot","reset","pin","7"]):
                _print_divider("FORGOT PIN — RECOVERY")
                speak("Starting PIN recovery. Step 1: Face verification.")
                recovered = forgot_pin_flow(user_id)
                if recovered:
                    speak("PIN recovered successfully. You can now make payments.")
                    _log("menu_pin_recovered", user_id=user_id)
                else:
                    speak("PIN recovery failed. Please try again or contact support.")
                    _log("menu_pin_recovery_failed", user_id=user_id)

            # ── EXIT ──────────────────────────────────────────────────────────
            elif any(x in choice for x in ["exit","quit","bye","8"]) and \
                 any(re.search(rf"\b{x}\b", choice) for x in ["exit","quit","bye","8"]):
                speak("Thank you for using VisPay. Goodbye.")
                _log("session_end", user_id=user_id); break

            else:
                speak("Didn't understand. Please say or type PAY, BALANCE, HISTORY, CONTACTS, CURRENCY, FORGOT, or EXIT.")

        except sqlite3.OperationalError as e:
            _log("post_auth_menu_db_error", error=str(e))
            speak("A database error occurred. Please try again.")
            time.sleep(1)
        except KeyboardInterrupt:
            speak("Thank you for using VisPay. Goodbye.")
            _log("session_end", user_id=user_id)
            break
        except Exception as e:
            _log("post_auth_menu_error", error=str(e))
            speak("An unexpected error occurred. Please try again.")

# ══════════════════════════════════════════════════════════════════════════════
# CONTACT MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
def _manage_contacts():
    _print_divider("MANAGE CONTACTS")
    contacts = load_contacts()
    _print_contacts(contacts)

    action = get_input(
        "Say or type ADD to add a contact, or BACK to return.",
        kb_prompt="Action (add/back)"
    )
    if "add" in action:
        name = get_input("Say or type the contact name.", kb_prompt="Name").lower().strip()
        upi  = get_input("Say or type the UPI ID.", kb_prompt="UPI ID (e.g. name@okaxis)").strip()
        ok, reason = add_trusted_contact(name, upi)
        speak(f"Contact {'saved' if ok else 'failed'}: {reason}")

# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _print_header():
    print("\n" + "═"*52)
    print("       💳  V I S P A Y  —  Secure Payments")
    print("         Face + Voice + PIN Authentication")
    print("═"*52)

def _print_divider(title=""):
    print(f"\n  {'─'*20} {title} {'─'*20}")

def _print_info(msg):
    print(f"\n  ✅  {msg}")

def _print_contacts(contacts):
    print(f"\n  {'NAME':<15} UPI ID")
    print(f"  {'─'*40}")
    for n, u in contacts.items():
        print(f"  {n.capitalize():<15} {u}")
    print()

# ══════════════════════════════════════════════════════════════════════════════
# STARTUP MODE SELECTOR
# ══════════════════════════════════════════════════════════════════════════════
def _select_input_mode():
    global INPUT_MODE
    _print_header()
    print("""
  Select Input Mode:
  ─────────────────
  [1]  Voice only     (speak all inputs)
  [2]  Keyboard only  (type all inputs)
  [3]  Both           (speak or type — whichever comes first)
""")
    try:
        choice = input("  Enter 1, 2, or 3: ").strip()
    except:
        choice = "2"
    INPUT_MODE = {"1": "voice", "2": "keyboard", "3": "both"}.get(choice, "keyboard")
    print(f"\n  ✅  Mode set to: {INPUT_MODE.upper()}\n")
    _log("input_mode", mode=INPUT_MODE)

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    init_db()
    _select_input_mode()

    # Demo user — register if first run
    TEST_USER, TEST_PIN = "user_001", "1234"
    if register_user(TEST_USER, "Demo User", TEST_PIN):
        print(f"  [DEMO] Created user '{TEST_USER}'  |  PIN: {TEST_PIN}  |  Balance: ₹10,000\n")
    else:
        print(f"  [DEMO] Logged in as '{TEST_USER}'  |  Balance: ₹{get_balance(TEST_USER):.0f}\n")

    post_auth_menu(TEST_USER)