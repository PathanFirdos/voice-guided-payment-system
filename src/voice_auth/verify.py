"""
verify.py — Voice verification for VisPay
No name needed. Matches voice against ALL enrolled users automatically.
"""

import os, sys, time, pickle, numpy as np, sounddevice as sd

import torchaudio
if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda *a, **k: None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from tts import speak
from user_manager import list_enrolled

SAMPLE_RATE   = 16000
DURATION      = 5
SAVE_DIR      = os.path.join(BASE_DIR, "enrolled_voices")
THRESHOLD     = 0.72
MAX_ATTEMPTS  = 3
PASS_REQUIRED = 2


def log(msg): print(f"\n  >>> {msg}")


def load_all_embeddings() -> dict:
    """Load embeddings for every enrolled user."""
    enrolled = list_enrolled()
    if not enrolled:
        log("No users enrolled. Please run record.py first.")
        speak("No users enrolled. Please enroll first.", force=True)
        sys.exit(1)

    embeddings = {}
    for uid in enrolled:
        path = os.path.join(SAVE_DIR, uid, "voice_embed.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                embeddings[uid] = pickle.load(f)

    log(f"Loaded {len(embeddings)} enrolled voice(s): {list(embeddings.keys())}")
    return embeddings


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


# In voice_auth/verify.py, replace the record_audio() function:
def record_audio() -> np.ndarray:
    print("  [mic] Recording...", end=" ", flush=True)
    try:
        sd.stop()                          # ← reset any stale PortAudio state
        time.sleep(0.1)                    # ← brief settle
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocking=False                 # explicit non-blocking
        )
        sd.wait()
        print("done.")
        return np.squeeze(audio)
    except Exception as e:
        print(f"\n  [mic] Recording failed: {e}")
        return np.zeros(DURATION * SAMPLE_RATE, dtype="float32")


def extract_embedding(model, audio) -> np.ndarray:
    import torch
    sig = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return model.encode_batch(sig).squeeze().cpu().numpy()


def cosine_similarity(a, b) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))


def best_match(test_emb: np.ndarray, all_embeddings: dict):
    """Return (best_user_id, best_score) across all enrolled users."""
    best_uid   = None
    best_score = -1.0
    for uid, emb in all_embeddings.items():
        score = cosine_similarity(test_emb, emb)
        if score > best_score:
            best_score = score
            best_uid   = uid
    return best_uid, best_score


def verify() -> None:
    all_embeddings = load_all_embeddings()
    model          = load_model()

    # ── instruct ONCE ─────────────────────────────────────────────────
    log(f"Starting verification — {MAX_ATTEMPTS} recordings.")
    log("Say clearly:  My voice is my secure password")
    log(f"Need {PASS_REQUIRED} of {MAX_ATTEMPTS} matches to grant access.")
    print()
    speak(f"Say: My voice is my secure password, {MAX_ATTEMPTS} times.", force=True)
    time.sleep(1.0)

    # ── attempt loop ──────────────────────────────────────────────────
    passed      = 0
    scores      = []
    matched_ids = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        log(f"Attempt {attempt} of {MAX_ATTEMPTS} — speak now.")
        speak(f"Attempt {attempt}.", force=True)

        audio    = record_audio()
        test_emb = extract_embedding(model, audio)

        uid, score = best_match(test_emb, all_embeddings)
        scores.append(score)
        matched_ids.append(uid if score >= THRESHOLD else None)

        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        status = "PASS ✓" if score >= THRESHOLD else "FAIL ✗"
        print(f"  [score] [{bar}] {score:.4f}  {status}  best match → {uid}")

        if score >= THRESHOLD:
            passed += 1
            speak("Match.", force=True)
        else:
            speak("No match." if attempt == MAX_ATTEMPTS else "No match. Try again.", force=True)

        if attempt < MAX_ATTEMPTS:
            time.sleep(2.0)  

    # ── final decision ────────────────────────────────────────────────
    avg   = np.mean(scores)
    # majority voted user
    valid = [u for u in matched_ids if u is not None]
    final_user = max(set(valid), key=valid.count) if valid else None

    print(f"\n  {'─'*46}")
    print(f"  Scores   : {[f'{s:.3f}' for s in scores]}")
    print(f"  Average  : {avg:.4f}")
    print(f"  Passed   : {passed}/{MAX_ATTEMPTS}  (need {PASS_REQUIRED})")
    print(f"  Identity : {final_user or 'unknown'}")
    print(f"  {'─'*46}")

    if passed >= PASS_REQUIRED and final_user:
        print(f"\n  ✓  ACCESS GRANTED  —  {final_user}\n")
        speak(f"Access granted. Welcome {final_user}.", force=True)
        # Write matched user_id for main.py to read across subprocess boundary
        try:
            import tempfile
            _handshake = os.path.join(tempfile.gettempdir(), "vispay_last_user.txt")
            with open(_handshake, "w") as _f:
                _f.write(final_user)
        except Exception as _e:
            print(f"  [warn] Could not write handshake file: {_e}")
        sys.exit(0)
    else:
        print(f"\n  ✗  ACCESS DENIED\n")
        speak("Voice verification failed. Access denied.", force=True)
        sys.exit(1)


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  VisPay — Voice Verification")
    print("="*50)
    verify()