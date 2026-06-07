"""
preprocess.py — VisPay currency recognition (standalone).

Works without infer.py / confidence.py — all logic is self-contained.

Fixes vs original:
  1. Blocking speak() removed → from tts import speak (non-blocking daemon thread)
  2. speak() inside camera loop froze the frame — non-blocking call keeps loop running
  3. Model loaded at module level — crashes on import if .tflite missing.
     Now loaded lazily inside run_recognition() with a clear error message.
  4. Sliding window used list slice (predictions[-N:]) but never reset when
     the predicted class changed, so stale votes carried over to the new class.
     Fixed with a proper window that resets on class change.
  5. "Background" was announced as "Background rupees note detected" — silenced.
  6. No on-screen stability progress bar — added.
  7. Camera hung when speak() blocked — fixed by non-blocking TTS.
  8. No cooldown after announcement — same note re-announced every frame once
     stable. Added cooldown_frames to suppress repeats.
  9. No post-detection menu — added POST_DETECT_MENU phase with voice + key input.
"""

import cv2
import numpy as np
import os
import sys
import time
import threading

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from tts import speak, speak_and_wait

# ── Settings ──────────────────────────────────────────────────────────────────
MODEL_PATH      = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "currency_recognition", "models", "currency",
                               "VisPay_currency_model.tflite")
CURRENCY_MAP    = {
    0: "10", 1: "100", 2: "20",  3: "200",
    4: "2000", 5: "50", 6: "500", 7: "Background"
}
INPUT_SIZE      = (224, 224)
WINDOW_NAME     = "VisPay — Currency Recognition"
BAR_HEIGHT      = 24
EXIT_KEY        = 27      # ESC

# Smoothing / announcement settings
WINDOW_SIZE     = 8       # frames in sliding vote window
MIN_PROB        = 0.50    # minimum instantaneous prob to count a frame
VOTE_FRACTION   = 0.75    # fraction of window that must agree (6 of 8)
COOLDOWN_FRAMES = 30      # frames to suppress same-label repeat

# Post-detection menu settings
MENU_TIMEOUT    = 10.0    # seconds to wait for user input before re-scanning


# ── Lazy model loader ─────────────────────────────────────────────────────────
_interpreter    = None
_input_details  = None
_output_details = None

def _load_model():
    global _interpreter, _input_details, _output_details
    try:
        import tensorflow as tf
    except ImportError:
        print("[currency] TensorFlow not installed. Run: pip install tensorflow")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"[currency] Model not found: {MODEL_PATH}")
        print("  Place VisPay_currency_model.tflite at that path and retry.")
        sys.exit(1)

    _interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    _interpreter.allocate_tensors()
    _input_details  = _interpreter.get_input_details()
    _output_details = _interpreter.get_output_details()
    print(f"[currency] Model loaded.")


# ── Inference (single frame) ──────────────────────────────────────────────────
def _predict(frame_bgr):
    """Returns (class_id, label, probability)."""
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = cv2.resize(rgb, INPUT_SIZE).astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)

    _interpreter.set_tensor(_input_details[0]['index'], tensor)
    _interpreter.invoke()
    probs    = _interpreter.get_tensor(_output_details[0]['index'])[0]
    class_id = int(np.argmax(probs))
    return class_id, CURRENCY_MAP.get(class_id, "?"), float(probs[class_id])


# ── HUD ───────────────────────────────────────────────────────────────────────
def _draw_hud(frame, label, prob, fill, announced):
    h, w = frame.shape[:2]
    color = (0, 255, 0) if announced else (0, 200, 255)

    if label != "Background":
        cv2.putText(frame, f"{label} rupees  ({prob:.0%})",
                    (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
    else:
        cv2.putText(frame, "No note detected",
                    (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)

    # Stability bar
    bar_y = h - BAR_HEIGHT
    cv2.rectangle(frame, (0, bar_y), (w, h), (30, 30, 30), -1)
    bar_color = (0, 220, 0) if fill >= 1.0 else (0, 160, 255)
    cv2.rectangle(frame, (0, bar_y), (int(w * fill), h), bar_color, -1)
    cv2.putText(frame, "Stability",
                (8, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "ESC to exit",
                (w - 140, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)


def _draw_menu_hud(frame, detected_label, elapsed, timeout):
    """HUD overlay shown during the post-detection menu."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(frame, f"Detected: {detected_label} rupees",
                (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 120), 2)
    cv2.putText(frame, "1 / 'again'  — Scan another note",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    cv2.putText(frame, "2 / 'pay'    — Proceed to payment",
                (20, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    cv2.putText(frame, "3 / 'exit'   — End session",
                (20, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    # Countdown bar
    remaining = max(0.0, timeout - elapsed)
    frac = remaining / timeout
    bar_y = h - BAR_HEIGHT
    cv2.rectangle(frame, (0, bar_y), (w, h), (20, 20, 20), -1)
    bar_color = (0, 200, 255) if frac > 0.3 else (0, 80, 255)
    cv2.rectangle(frame, (0, bar_y), (int(w * frac), h), bar_color, -1)
    cv2.putText(frame, f"Auto-scan in {remaining:.0f}s",
                (8, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


# ── Voice input (non-blocking, result written to a list) ─────────────────────
def _listen_voice(result_holder: list, keywords: dict, cancel: threading.Event):
    """
    Runs on a daemon thread. Listens for one utterance and maps it to a
    choice integer via `keywords` dict  e.g. {"again": 1, "pay": 2, "exit": 3}.
    Writes the integer to result_holder[0] when recognised, or 0 on failure.
    Does nothing if cancel is set before recognition completes (keyboard won).
    """
    try:
        import speech_recognition as sr
        r   = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            r.adjust_for_ambient_noise(source, duration=0.3)
            audio = r.listen(source, timeout=8, phrase_time_limit=4)

        if cancel.is_set():
            return   # keyboard already handled it — discard result

        text = r.recognize_google(audio).lower()
        print(f"[menu] Heard: {text!r}")

        if cancel.is_set():
            return

        for kw, choice in keywords.items():
            if kw in text:
                result_holder[0] = choice
                return
        result_holder[0] = 0   # heard something but no keyword matched
    except Exception as e:
        if not cancel.is_set():
            print(f"[menu] Voice listen failed: {e}")
            result_holder[0] = 0


def _start_listener(result_holder, keywords, cancel):
    """Spawn a fresh voice listener daemon thread."""
    t = threading.Thread(
        target=_listen_voice,
        args=(result_holder, keywords, cancel),
        daemon=True,
    )
    t.start()
    return t


# ── Post-detection menu ───────────────────────────────────────────────────────
def _post_detect_menu(cap, detected_label: str) -> str:
    """
    Freeze the camera feed and show an overlay menu.
    Returns one of: 'again' | 'pay' | 'exit'

    Input accepted simultaneously — whichever arrives first wins:
      Voice: "again" / "scan"   → scan again
             "pay"  / "payment" → proceed to payment
             "exit" / "quit"    → end session
      Key:   1 → again   2 → pay   3 / ESC → exit

    When keyboard wins, a cancel_event is set so the voice thread
    discards its result and exits cleanly — no ghost writes after return.
    """
    voice_keywords = {
        "again": 1, "scan": 1, "rescan": 1,
        "pay": 2, "payment": 2, "proceed": 2,
        "exit": 3, "quit": 3, "end": 3,
    }
    outcome_map = {1: "again", 2: "pay", 3: "exit"}

    # Announce the menu once (blocking so the user hears it fully)
    speak_and_wait(
        f"{detected_label} rupees detected. "
        "Press 1 or say again to scan another note. "
        "Press 2 or say pay to proceed to payment. "
        "Press 3 or say exit to end the session."
    )

    # Shared state between keyboard poller and voice thread
    voice_result  = [None]          # None = still listening
    cancel_event  = threading.Event()

    _start_listener(voice_result, voice_keywords, cancel_event)

    start = time.monotonic()
    ret, frozen = cap.read()        # snapshot frame to display while menu is up
    if not ret:
        frozen = np.zeros((480, 640, 3), dtype=np.uint8)

    repromed = False

    def _keyboard_win(outcome: str) -> str:
        """Set cancel so voice thread exits cleanly, then return outcome."""
        cancel_event.set()
        print(f"[menu] Key choice: {outcome}")
        return outcome

    while True:
        elapsed = time.monotonic() - start
        display = frozen.copy()
        _draw_menu_hud(display, detected_label, elapsed, MENU_TIMEOUT)
        cv2.imshow(WINDOW_NAME, display)

        # ── Keyboard (checked every 40 ms) ────────────────────────────────────
        key = cv2.waitKey(40) & 0xFF
        if key == ord('1'):
            return _keyboard_win("again")
        if key == ord('2'):
            return _keyboard_win("pay")
        if key in (ord('3'), EXIT_KEY):
            return _keyboard_win("exit")

        # ── Voice result ──────────────────────────────────────────────────────
        if voice_result[0] is not None:
            choice = voice_result[0]
            if choice in outcome_map:
                print(f"[menu] Voice choice: {outcome_map[choice]}")
                return outcome_map[choice]
            else:
                # Heard something but no keyword matched — re-prompt once
                if not repromed:
                    speak(
                        "Sorry, didn't catch that. Say again, pay, or exit.",
                        interrupt=False,
                    )
                    repromed      = True
                    voice_result  = [None]
                    cancel_event  = threading.Event()   # fresh event for new listener
                    _start_listener(voice_result, voice_keywords, cancel_event)

        # ── Timeout ───────────────────────────────────────────────────────────
        if elapsed >= MENU_TIMEOUT:
            cancel_event.set()      # shut down any still-listening thread
            speak(
                "No input received. Returning to scanner." if not repromed
                else "Returning to scanner.",
                interrupt=False,
            )
            return "again"


# ── Main loop ─────────────────────────────────────────────────────────────────
def run_recognition():
    _load_model()
    speak_and_wait("Currency recognition started. Please hold a note in front of the camera.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak_and_wait("Camera could not be opened.")
        return

    # Sliding window state
    window         = []
    last_class     = -1
    last_announced = ""
    cooldown_left  = 0

    # FSM phase: 'scanning' | 'menu'
    phase          = "scanning"
    detected_label = ""

    print("[currency] Running — press ESC to stop.")

    while True:
        # ── SCANNING phase ────────────────────────────────────────────────────
        if phase == "scanning":
            ret, frame = cap.read()
            if not ret:
                break

            class_id, label, prob = _predict(frame)

            if cooldown_left > 0:
                cooldown_left -= 1

            # Window update — reset on class change
            if prob >= MIN_PROB:
                if class_id != last_class:
                    window     = []
                    last_class = class_id
                window.append(class_id)
                if len(window) > WINDOW_SIZE:
                    window.pop(0)

            fill         = len(window) / WINDOW_SIZE
            announced    = False
            votes_needed = int(WINDOW_SIZE * VOTE_FRACTION)
            stable       = (len(window) == WINDOW_SIZE and
                            window.count(class_id) >= votes_needed)

            if stable and label != "Background":
                if not (label == last_announced and cooldown_left > 0):
                    # Note confirmed — transition to menu
                    last_announced = label
                    cooldown_left  = COOLDOWN_FRAMES
                    announced      = True
                    detected_label = label
                    phase          = "menu"

            _draw_hud(frame, label, prob, fill, announced)
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == EXIT_KEY:
                break

        # ── MENU phase ────────────────────────────────────────────────────────
        elif phase == "menu":
            # Reset window so next scan starts fresh
            window     = []
            last_class = -1

            choice = _post_detect_menu(cap, detected_label)
            print(f"[menu] Choice: {choice}")

            if choice == "again":
                phase = "scanning"
                speak("Scanning for next note.", interrupt=False)

            elif choice == "pay":
                speak_and_wait("Proceeding to payment.")
                # ── Hook your TRANSACTION / payment flow here ──────────────
                _handle_payment(detected_label)
                # After payment, return to scanning
                phase = "scanning"
                speak("Ready to scan another note.", interrupt=False)

            elif choice == "exit":
                break

    cap.release()
    cv2.destroyAllWindows()
    speak_and_wait("Currency recognition ended. Goodbye.")
    print("[currency] Session ended.")


# ── Payment stub — replace with your real transaction flow ────────────────────
def _handle_payment(label: str):
    """
    Placeholder called when the user chooses 'pay'.
    Replace the body with your actual TRANSACTION FSM state / payment logic.
    """
    speak_and_wait(f"Processing payment for {label} rupees. Please wait.")
    print(f"[payment] TODO: implement transaction for ₹{label}")
    time.sleep(1.5)   # simulate processing
    speak_and_wait("Payment complete. Thank you.")


if __name__ == "__main__":
    run_recognition()