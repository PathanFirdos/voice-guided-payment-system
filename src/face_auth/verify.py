"""
verify.py — Face verification for VisPay.

Fixes applied vs previous version:
  1. (NEW) TTS BLEED — speak() is non-blocking; the old code called
     speak("Please say your enrolled name") then immediately called
     r.listen(), so the microphone captured the tail of the TTS audio
     itself and SR returned "please_say_your_enrolled_name" as the
     user ID.  Fixed by replacing speak() + time.sleep(2.0) with
     speak_and_wait() followed by a 0.6 s post-TTS silence gap so the
     mic opens only after the speaker has gone fully quiet.

  2. (NEW) EDGE-TTS SHUTDOWN CRASH — "cannot schedule new futures after
     interpreter shutdown" fires when sys.exit() tears down the process
     while a daemon TTS thread is still alive.  Fixed by calling
     tts.wait_until_done() before every sys.exit() path, and importing
     wait_until_done from tts.

  3. Removed duplicate speak() — now imported from tts.py.
  4. load_embeddings() now reads stored 160×160 crops directly (no
     redundant MTCNN pass); fallback kept for full-frame images.
  5. Negative / out-of-bounds MTCNN box coords clamped.
  6. frame_count incremented every frame (face-absent = miss), not only
     on face-detected frames.
  7. cv2.imshow called before the break-on-frame_count check so the
     last annotated frame is always displayed.
  8. DATASET_DIR anchored to __file__ so the module resolves correctly
     regardless of the caller's working directory.
"""

import cv2
import os
import sys
import numpy as np
import time
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
from tts import speak, speak_and_wait, wait_until_done   # FIX 2: import wait_until_done

# ── Settings ──────────────────────────────────────────────────────────────────
_HERE         = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR   = os.path.join(_HERE, "enrolled_faces")   # absolute path
THRESHOLD     = 0.65    # higher = stricter (cosine similarity, range 0–1)
VERIFY_FRAMES = 10      # total frames to sample

# Post-TTS silence buffer — wait this long after TTS finishes before
# opening the microphone.  Prevents speaker echo being captured as input.
POST_TTS_SILENCE = 0.6  # seconds  (FIX 1)

# ── Init (module-level, load once) ────────────────────────────────────────────
detector = MTCNN()
embedder = FaceNet()


# ── Voice input ───────────────────────────────────────────────────────────────
def get_user_id_by_voice() -> str:
    """
    Ask the user to say their enrolled ID.

    FIX 1: Uses speak_and_wait() so the function blocks until TTS audio
    is fully finished, then waits POST_TTS_SILENCE seconds of quiet before
    opening the microphone.  This prevents the SR engine from transcribing
    the TTS prompt itself (the old bug that produced user IDs like
    'please_say_your_android').
    """
    import speech_recognition as sr

    r = sr.Recognizer()
    mic_index = None

    for i, _name in enumerate(sr.Microphone.list_microphone_names()):
        try:
            with sr.Microphone(device_index=i):
                mic_index = i
                break
        except Exception:
            continue

    if mic_index is None:
        speak_and_wait("No working microphone detected. Please type your user ID.")
        return input("User ID: ").strip().lower().replace(" ", "_")

    try:
        # FIX 1: speak_and_wait blocks until audio is done, then we add
        # a silence gap before the mic opens — no more TTS bleed.
        speak_and_wait("Please say your enrolled name clearly.")
        time.sleep(POST_TTS_SILENCE)   # ← silence gap; mic opens here

        with sr.Microphone(device_index=mic_index) as source:
            r.adjust_for_ambient_noise(source, duration=0.4)
            print("  [face] Listening for user ID...")
            audio = r.listen(source, timeout=7, phrase_time_limit=4)

        user_id = r.recognize_google(audio).replace(" ", "_").lower()
        speak_and_wait(f"I heard {user_id}.")
        return user_id

    except sr.WaitTimeoutError:
        print("  [face] No speech detected within timeout.")
        speak_and_wait("No speech detected. Please type your user ID.")
        return input("User ID: ").strip().lower().replace(" ", "_")
    except Exception as e:
        print(f"  [face] Voice input failed: {e}")
        speak_and_wait("Voice input failed. Please type your user ID.")
        return input("User ID: ").strip().lower().replace(" ", "_")


# ── Load stored embeddings ────────────────────────────────────────────────────
def load_embeddings(user_id: str) -> np.ndarray | None:
    """
    Stored images are already 160×160 crops from enroll.py.
    Read them directly without re-running MTCNN on each one.
    Falls back to a full MTCNN pass only for images larger than 200 px.
    DATASET_DIR is absolute (anchored to __file__).
    """
    user_path = os.path.join(DATASET_DIR, user_id)
    print(f"  [face] load_embeddings: looking in {user_path}")

    if not os.path.exists(user_path):
        print(f"  [face] Folder not found: {user_path}")
        return None

    embeddings = []
    for fname in sorted(os.listdir(user_path)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(user_path, fname)
        img = cv2.imread(path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        if w <= 200 and h <= 200:
            # Already a 160×160 crop — embed directly
            face = cv2.resize(rgb, (160, 160))
        else:
            # Full-frame image — detect face first
            faces = detector.detect_faces(rgb)
            if not faces:
                continue
            fx, fy, fw, fh = faces[0]['box']
            fx, fy = max(0, fx), max(0, fy)
            fw = min(w - fx, fw)
            fh = min(h - fy, fh)
            if fw <= 0 or fh <= 0:
                continue
            face = cv2.resize(rgb[fy:fy + fh, fx:fx + fw], (160, 160))

        emb = embedder.embeddings([face])[0]
        embeddings.append(emb)

    print(f"  [face] Loaded {len(embeddings)} embeddings for '{user_id}'")
    return np.array(embeddings) if embeddings else None


# ── Main verification ─────────────────────────────────────────────────────────
def run_verification() -> str:
    """
    Returns "VERIFIED", "FAILED", or "NO_ENROLLMENT".

    FIX 2: every early-exit path calls wait_until_done() before returning
    so that TTS daemon threads finish cleanly and don't trigger the
    'cannot schedule new futures after interpreter shutdown' error.
    """
    user_id = get_user_id_by_voice()
    stored_embeddings = load_embeddings(user_id)

    if stored_embeddings is None or len(stored_embeddings) == 0:
        msg = f"No enrollment found for {user_id}. Please enroll first."
        print(msg)
        speak_and_wait(msg)          # FIX 2: blocking — audio finishes before return
        return "NO_ENROLLMENT"

    speak_and_wait("Please look at the camera for face verification.")

    cap = cv2.VideoCapture(0)
    match_count = 0
    frame_count = 0

    while frame_count < VERIFY_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)

        if faces:
            x, y, w, h = faces[0]['box']
            x, y = max(0, x), max(0, y)
            w    = min(frame.shape[1] - x, w)
            h    = min(frame.shape[0] - y, h)

            if w > 0 and h > 0:
                face_crop = cv2.resize(rgb[y:y + h, x:x + w], (160, 160))
                emb       = embedder.embeddings([face_crop])[0]
                sims      = cosine_similarity([emb], stored_embeddings)
                score     = float(np.max(sims))

                color = (0, 255, 0) if score >= THRESHOLD else (0, 100, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"Score: {score:.2f}  ({frame_count+1}/{VERIFY_FRAMES})",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if score >= THRESHOLD:
                    match_count += 1
        else:
            speak("No face detected", interrupt=False)
            cv2.putText(frame, "No face", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Face Verification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # ── Decision ──────────────────────────────────────────────────────────────
    required = VERIFY_FRAMES // 2
    if match_count >= required:
        msg = "Face verified successfully. Payment authorized."
        print(f"VERIFIED  ({match_count}/{frame_count} frames matched)")
        speak_and_wait(msg)          # FIX 2: blocking before return
        return "VERIFIED"
    else:
        msg = "Face verification failed. Switching to voice authentication."
        print(f"FAILED  ({match_count}/{frame_count} frames matched, needed {required})")
        speak_and_wait(msg)          # FIX 2: blocking before return
        return "FAILED"


if __name__ == "__main__":
    result = run_verification()
    wait_until_done()                # FIX 2: drain TTS before process exits
    sys.exit(0 if result == "VERIFIED" else 1)