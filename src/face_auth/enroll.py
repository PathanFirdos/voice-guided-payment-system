"""
enroll.py — Face enrollment for VisPay.

Fixes in this version:
  1. CAPTURE THROTTLE — minimum 0.4s gap between saved frames so 50 images
     takes at least 20 seconds and captures genuinely different poses.
  2. FRAME COUNTER always rendered correctly — drawn on every frame, not
     only on save frames, so display never jumps or skips.
  3. COMPLETION ANNOUNCEMENT — speaks "X images captured" and waits for
     TTS to finish before closing the window (join the audio thread).
  4. EXIT FLOW — window stays open on the final frame long enough for the
     user to see "50/50" before it closes.
  5. All previous fixes retained (non-blocking TTS, coord clamping, arrow
     directions, mouth averaging, stale-image clearing).
"""

import cv2
import os
import time
import threading
import numpy as np
from mtcnn import MTCNN
import sys, os as _os
_SRC = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from tts import speak, speak_and_wait
from user_manager import register_user
# ── Settings ──────────────────────────────────────────────────────────────────
SAVE_DIR            = "enrolled_faces"
NUM_IMAGES          = 50
FRAMES_PER_ANGLE    = 8
INSTRUCTION_INTERVAL = 3.0   # seconds between repeated voice reminders
CAPTURE_INTERVAL    = 0.4    # FIX 1: minimum seconds between saved frames
BAR_HEIGHT          = 20

# ── Init ──────────────────────────────────────────────────────────────────────
os.makedirs(SAVE_DIR, exist_ok=True)
detector = MTCNN()


# ── Voice name input ──────────────────────────────────────────────────────────
def get_user_name():
    import speech_recognition as sr

    r         = sr.Recognizer()
    mic_index = None
    name      = None

    # ── Try voice input first ─────────────────────────────────────────────────
    for i, _n in enumerate(sr.Microphone.list_microphone_names()):
        try:
            with sr.Microphone(device_index=i):
                mic_index = i
                break
        except Exception:
            continue

    if mic_index is not None:
        try:
            with sr.Microphone(device_index=mic_index) as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
                speak_and_wait("Please say your user ID clearly after the beep.")
                time.sleep(0.5)
                audio = r.listen(source, timeout=5)
            heard = r.recognize_google(audio).replace(" ", "_").lower()
            speak_and_wait(f"I heard {heard}. Is that correct? Type Y to confirm or N to type manually.")
            confirm = input("  Confirm (Y/N): ").strip().lower()
            if confirm == "y":
                name = heard
        except Exception as e:
            print(f"  Voice input failed: {e}")

    # ── Keyboard fallback (always shown if voice failed or rejected) ──────────
    if not name:
        speak_and_wait("Please type your user ID and press Enter.")
        print("\n  ┌─────────────────────────────────────────┐")
        print("  │  Enter the same ID used in VisPay login  │")
        print("  │  Example:  user_001                      │")
        print("  └─────────────────────────────────────────┘")
        while True:
            raw = input("  User ID: ").strip().lower().replace(" ", "_")
            if raw:
                name = raw
                break
            print("  Cannot be empty. Try again.")

    speak_and_wait(f"Enrolling face for {name}.")
    print(f"\n  [enroll] User ID: {name}")

    user_dir = os.path.join(SAVE_DIR, name)
    os.makedirs(user_dir, exist_ok=True)

    # Clear stale images so re-enrollment starts fresh
    stale = [f for f in os.listdir(user_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if stale:
        print(f"  Clearing {len(stale)} old images...")
        for f in stale:
            os.remove(os.path.join(user_dir, f))

    return name, user_dir

    # Clear stale images so re-enrollment starts clean
    for f in os.listdir(user_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            os.remove(os.path.join(user_dir, f))

    return name, user_dir


# ── Face coverage check ───────────────────────────────────────────────────────
def check_face_coverage(face):
    x, y, w, h = face['box']
    kp = face['keypoints']
    mouth_y = (kp['mouth_left'][1] + kp['mouth_right'][1]) / 2

    coverage = {
        "left":  kp['left_eye'][0]  - x       > w * 0.15,
        "right": (x + w) - kp['right_eye'][0] > w * 0.15,
        "up":    kp['nose'][1]      - y        > h * 0.15,
        "down":  (y + h)            - mouth_y  > h * 0.15,
    }

    missing = [k for k, v in coverage.items() if not v]
    ratio   = 1 - len(missing) / 4
    return missing, ratio


# ── Draw helpers ──────────────────────────────────────────────────────────────
def draw_arrows(frame, missing):
    fh, fw = frame.shape[:2]
    c, t, L = (0, 0, 255), 3, 60
    if "left"  in missing: cv2.arrowedLine(frame, (fw-40, fh//2), (fw-40-L, fh//2), c, t)
    if "right" in missing: cv2.arrowedLine(frame, (40,    fh//2), (40+L,    fh//2), c, t)
    if "up"    in missing: cv2.arrowedLine(frame, (fw//2, fh-40), (fw//2, fh-40-L), c, t)
    if "down"  in missing: cv2.arrowedLine(frame, (fw//2,    40), (fw//2,    40+L), c, t)


def draw_hud(frame, count, coverage_ratio, instruction):
    """FIX 2: draw counter + bar + instruction on EVERY frame."""
    fh, fw = frame.shape[:2]

    # Progress bar at bottom
    cv2.rectangle(frame, (0, fh - BAR_HEIGHT), (fw, fh), (40, 40, 40), -1)
    cv2.rectangle(frame, (0, fh - BAR_HEIGHT),
                  (int(fw * count / NUM_IMAGES), fh), (0, 200, 80), -1)

    # Counter — large, always visible
    label = f"{count} / {NUM_IMAGES}"
    cv2.putText(frame, label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Coverage percentage
    cv2.putText(frame, f"Coverage {int(coverage_ratio * 100)}%",
                (10, fh - BAR_HEIGHT - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Current instruction at top-right
    cv2.putText(frame, instruction, (fw - 360, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 80), 2)


# ── Main enrollment ───────────────────────────────────────────────────────────
def run_enrollment():
    name, user_dir = get_user_name()
    cap = cv2.VideoCapture(0)

    instructions = [
        "Face forward",
        "Turn slightly left",
        "Turn slightly right",
        "Look up a little",
        "Look down a little",
    ]

    inst_index            = 0
    count                 = 0
    valid_frames          = 0
    last_instruction_time = 0.0
    last_capture_time     = 0.0   # FIX 1: throttle timestamp
    coverage_ratio        = 0.0

    speak(instructions[inst_index], force=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)

        # Periodic voice reminder — low priority, never cuts an instruction
        if (now - last_instruction_time > INSTRUCTION_INTERVAL
                and valid_frames < FRAMES_PER_ANGLE):
            speak(instructions[inst_index], interrupt=False)
            last_instruction_time = now

        if faces:
            face = faces[0]
            x, y, w, h = face['box']
            x = max(0, x);  y = max(0, y)
            w = min(frame.shape[1] - x, w)
            h = min(frame.shape[0] - y, h)

            if w > 0 and h > 0:
                missing, coverage_ratio = check_face_coverage(face)

                if missing:
                    draw_arrows(frame, missing)
                    speak("Please adjust your face position")
                    valid_frames = 0

                else:
                    # FIX 1: only save if enough time has passed
                    if now - last_capture_time >= CAPTURE_INTERVAL:
                        crop = cv2.resize(frame[y:y+h, x:x+w], (160, 160))
                        cv2.imwrite(os.path.join(user_dir, f"{count}.jpg"), crop)
                        count += 1
                        valid_frames     += 1
                        last_capture_time = now

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # Advance to next angle — wait for full instruction before continuing
                    if valid_frames >= FRAMES_PER_ANGLE and inst_index < len(instructions) - 1:
                        inst_index           += 1
                        valid_frames          = 0
                        last_instruction_time = 0.0
                        speak_and_wait(instructions[inst_index])
        else:
            coverage_ratio = 0.0
            speak("No face detected, please come in front of the camera")

        # FIX 2: HUD drawn every frame with correct count
        draw_hud(frame, count, coverage_ratio, instructions[inst_index])
        cv2.imshow("Face Enrollment", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if count >= NUM_IMAGES:
            # FIX 4: show final frame with "50/50" for 1.5s before closing
            draw_hud(frame, count, coverage_ratio, "Done!")
            cv2.imshow("Face Enrollment", frame)
            cv2.waitKey(1500)
            break

    cap.release()
    cv2.destroyAllWindows()

    # Speak completion and wait for full audio before exiting
    msg = f"Enrollment complete. {count} images captured for {name}."
    print(msg)
    speak_and_wait(msg)
    # Register in face user index
    register_user(name, name=name, image_count=count)

    return name


if __name__ == "__main__":
    run_enrollment()