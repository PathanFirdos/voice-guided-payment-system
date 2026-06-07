"""
record.py — Voice enrollment for VisPay
No name needed. Press key or say ONE/TWO. System assigns ID.
"""

import os, sys, time, pickle, numpy as np, sounddevice as sd
from scipy.io.wavfile import write as wav_write

import torchaudio
if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda *a, **k: None

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tts import speak
from user_manager import assign_new_id, list_enrolled, enrolled_count, collect_and_confirm_pin, _hash_pin

SAMPLE_RATE = 16000
DURATION    = 5
SAVE_DIR    = os.path.join(BASE_DIR, "enrolled_voices")
NUM_SAMPLES = 3


def log(msg): print(f"\n  >>> {msg}")


def listen_for_word(timeout=6) -> str:
    import queue, json, vosk
    model_path = os.path.join(BASE_DIR, "vosk-model")
    if not os.path.exists(model_path):
        import speech_recognition as sr
        r = sr.Recognizer()
        try:
            with sr.Microphone() as src:
                r.adjust_for_ambient_noise(src, duration=0.4)
                audio = r.listen(src, timeout=timeout, phrase_time_limit=3)
            return r.recognize_google(audio).strip().lower()
        except Exception:
            return ""

    model = vosk.Model(model_path)
    rec   = vosk.KaldiRecognizer(model, SAMPLE_RATE)
    q     = queue.Queue()

    def cb(indata, frames, t, status): q.put(bytes(indata))

    heard    = ""
    deadline = time.time() + timeout
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=4000,
                           dtype="int16", channels=1, callback=cb):
        while time.time() < deadline:
            try:
                data = q.get(timeout=0.5)
            except queue.Empty:
                continue
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text   = result.get("text","").strip().lower()
                if text:
                    heard = text
                    break
        if not heard:
            heard = json.loads(rec.PartialResult()).get("partial","").strip().lower()
    return heard


def ask_choice(prompt_voice: str, prompt_log: str) -> str:
    """
    Ask ONE / TWO via voice.
    Also accepts keyboard: press 1 or 2 then Enter as fallback.
    """
    import threading
    speak(prompt_voice, force=True)
    log(prompt_log)
    log("Say ONE or TWO  —  or press 1 / 2 on keyboard and hit Enter.")

    result  = [None]
    kb_done = threading.Event()

    def kb_input():
        val = input("  [keyboard 1/2]: ").strip()
        result[0] = val
        kb_done.set()

    t = threading.Thread(target=kb_input, daemon=True)
    t.start()

    kb_done.wait(timeout=1.5)   # short window — prefer voice
    if result[0] in ("1", "2"):
        log(f"Keyboard input: '{result[0]}'")
        return result[0]

    heard = listen_for_word(timeout=6)
    log(f"Heard: '{heard}'")

    if "one" in heard or "1" in heard:
        return "1"
    if "two" in heard or "2" in heard:
        return "2"

    # nothing clear — wait for keyboard result
    kb_done.wait(timeout=8)
    if result[0] in ("1", "2"):
        return result[0]

    return "1"   # default to ONE (proceed)


def load_model():
    log("Loading voice model...")
    try:
        try:
            from speechbrain.pretrained import EncoderClassifier
        except ImportError:
            from speechbrain.inference.classifiers import EncoderClassifier
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(BASE_DIR, ".sb_cache"),
            run_opts={"device": "cpu"}
        )
        log("Model ready.")
        return model
    except Exception as e:
        import traceback; traceback.print_exc()
        sys.exit(2)


def record_audio() -> np.ndarray:
    print("  [mic] Recording...", end=" ", flush=True)
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    print("done.")
    return np.squeeze(audio)


def extract_embedding(model, audio):
    import torch
    sig = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return model.encode_batch(sig).squeeze().cpu().numpy()


def enroll() -> None:
    # ── how many enrolled so far ──────────────────────────────────────
    count = enrolled_count()
    log(f"Enrolled users so far: {count}")

    choice = ask_choice(
        prompt_voice="Say ONE to enroll a new voice. Say TWO to cancel.",
        prompt_log="ONE = enroll new user    TWO = cancel"
    )

    if choice == "2":
        log("Enrollment cancelled.")
        speak("Okay, cancelled.", force=True)
        sys.exit(0)

    # ── Step 1: Collect name (keyboard only — non-literate users get help) ──
    speak("Please type or say your name.", force=True)
    print("\n  Enter your name (press Enter to skip):")
    name_raw = input("  Name: ").strip()
    name = name_raw if name_raw else f"user_{int(time.time())}"
    speak(f"Hello {name}.", force=True)

    # ── Step 2: Voice recording ───────────────────────────────────────
    model = load_model()

    log(f"Starting enrollment — {NUM_SAMPLES} recordings.")
    log("Say clearly:  My voice is my secure password")
    print()
    speak(f"I will record your voice {NUM_SAMPLES} times.", force=True)
    speak("Each time say: My voice is my secure password.", force=True)
    speak("Sit quietly. Recording starts after each prompt.", force=True)
    time.sleep(1.5)

    embeddings = []
    tmp_dir    = os.path.join(SAVE_DIR, "_tmp_enroll")
    os.makedirs(tmp_dir, exist_ok=True)

    for i in range(NUM_SAMPLES):
        log(f"Recording {i+1} of {NUM_SAMPLES} — speak now.")
        speak(f"Recording {i+1}.", force=True)
        audio = record_audio()

        if i == 0:
            wav_write(os.path.join(tmp_dir, "voice.wav"),
                      SAMPLE_RATE, np.int16(audio * 32767))

        embeddings.append(extract_embedding(model, audio))

        if i < NUM_SAMPLES - 1:
            speak("Good.", force=True)
            time.sleep(1.0)

    # ── Step 3: Create PIN ────────────────────────────────────────────
    speak("Voice enrolled. Now let us create your secure PIN.", force=True)
    pin      = collect_and_confirm_pin()
    pin_hash = _hash_pin(pin)

    # ── Step 4: Register user with name + pin_hash ────────────────────
    user_id  = assign_new_id(name=name, pin_hash=pin_hash)
    user_dir = os.path.join(SAVE_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    embed_path = os.path.join(user_dir, "voice_embed.pkl")

    # Move temp files to final user dir
    import shutil
    for f in os.listdir(tmp_dir):
        shutil.move(os.path.join(tmp_dir, f), os.path.join(user_dir, f))
    os.rmdir(tmp_dir)

    final = np.mean(embeddings, axis=0)
    with open(embed_path, "wb") as f:
        pickle.dump(final, f)

    # ── Step 5: Confirm ───────────────────────────────────────────────
    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  ✅  ENROLLMENT COMPLETE              │")
    print(f"  │  User ID : {user_id:<26}│")
    print(f"  │  Name    : {name:<26}│")
    print(f"  │  PIN     : {'****':<26}│")
    print(f"  └─────────────────────────────────────┘\n")

    log(f"Enrolled: {user_id} | name={name}")
    speak(f"Registration complete. Your ID is {user_id}. Name is {name}. PIN saved.", force=True)


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  VisPay — Voice Enrollment")
    print("="*50)
    enroll()
    sys.exit(0)