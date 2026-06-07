"""
tts.py — VisPay unified TTS module
Place at: C:\\projects\\vispay\\src\\tts.py
Delete:   src/assistant/tts.py  src/face_auth/tts.py  src/voice_auth/tts.py

Import everywhere as:
    from tts import speak, speak_and_wait          # scripts in src/
    from ..tts import speak, speak_and_wait        # sub-packages

Engine priority:
  1. edge-tts  (neural quality, no API key, needs internet)
  2. gTTS      (fallback, also needs internet)
  3. Silent    (logs warning, never crashes the main loop)

pip install edge-tts gtts pygame
"""

import asyncio
import atexit
import os
import tempfile
import threading
import time
import logging

import pygame

logger = logging.getLogger(__name__)

# ── pygame init ────────────────────────────────────────────────────────────────
if not pygame.mixer.get_init():
    pygame.mixer.init()

# ── Voice map ──────────────────────────────────────────────────────────────────
# Best Microsoft neural voices for VisPay's market
_EDGE_VOICES: dict[str, str] = {
    "en": "en-IN-NeerjaNeural",   # Indian English — natural for VisPay users
    "hi": "hi-IN-SwaraNeural",    # Hindi
    "mr": "mr-IN-AarohiNeural",   # Marathi
}

# ── Cooldown / dedup state ─────────────────────────────────────────────────────
_last_text: str   = ""
_last_time: float = 0.0
_state_lock       = threading.Lock()
DEFAULT_COOLDOWN  = 2.5           # seconds before same phrase can repeat

# ── Single reusable audio thread ───────────────────────────────────────────────
_audio_thread: threading.Thread | None = None
_stop_event   = threading.Event()     # set to interrupt current playback


def _atexit_drain() -> None:
    """
    Join the TTS daemon thread before the interpreter shuts down.
    Prevents 'cannot schedule new futures after interpreter shutdown'
    from edge-tts when the process exits while audio is still playing.
    """
    _stop_event.set()
    if _audio_thread and _audio_thread.is_alive():
        _audio_thread.join(timeout=2.0)


atexit.register(_atexit_drain)


# ── Public API ─────────────────────────────────────────────────────────────────

def is_speaking() -> bool:
    """Return True if audio is currently playing on the background thread."""
    return _audio_thread is not None and _audio_thread.is_alive()


def wait_until_done(timeout: float = 10.0) -> None:
    """Block caller until the current audio finishes (or timeout elapses)."""
    if _audio_thread and _audio_thread.is_alive():
        _audio_thread.join(timeout=timeout)


def speak(
    text: str,
    lang: str = "en",
    force: bool = False,
    cooldown_secs: float = DEFAULT_COOLDOWN,
    interrupt: bool = True,
) -> None:
    """
    Speak *text* aloud — NON-BLOCKING.

    Returns immediately; audio plays on a daemon thread so camera loops
    (cap.read → MTCNN → imshow) never freeze waiting for speech to finish.

    Args:
        text:          Text to synthesise.
        lang:          Language code: 'en' (default), 'hi', 'mr'.
        force:         If True, bypass cooldown and always speak.
        cooldown_secs: Minimum gap before the same phrase repeats.
        interrupt:     If False and audio is already playing, skip this
                       phrase instead of cutting the current one off.
                       Use interrupt=False for low-priority ambient cues.
    """
    global _last_text, _last_time, _audio_thread

    if not text or not text.strip():
        return

    # Low-priority: skip rather than interrupt
    if not interrupt and is_speaking():
        return

    with _state_lock:
        now = time.time()
        if not force:
            if text == _last_text and (now - _last_time) < cooldown_secs:
                logger.debug("TTS cooldown — skipped: %r", text)
                return
        _last_text = text
        _last_time = now

    # Stop any currently playing audio (only when interrupt=True)
    _stop_event.set()
    if _audio_thread and _audio_thread.is_alive():
        _audio_thread.join(timeout=0.3)

    _audio_thread = threading.Thread(
        target=_audio_worker, args=(text, lang), daemon=True
    )
    _audio_thread.start()


def speak_and_wait(text: str, lang: str = "en", timeout: float = 15.0) -> None:
    """
    Speak *text* and BLOCK until audio finishes.

    Use this for critical instructions where the app must not proceed
    until the user has heard the full sentence — e.g. angle prompts
    during face enrollment, or 'authentication successful'.
    """
    speak(text, lang=lang, force=True, interrupt=True)
    wait_until_done(timeout=timeout)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _audio_worker(text: str, lang: str) -> None:
    """Runs on the daemon thread. Tries engines in order."""
    _stop_event.clear()
    if _speak_edge(text, lang):
        return
    if _stop_event.is_set():
        return
    if _speak_gtts(text, lang):
        return
    logger.warning("TTS: all engines failed for: %r", text)
    print(f"[TTS SILENT] {text}")


def _play_mp3(path: str) -> bool:
    """Play an mp3, honouring _stop_event for fast interruption."""
    try:
        # Re-init mixer on every call — safe from any thread
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            if _stop_event.is_set():
                pygame.mixer.music.stop()
                break
            time.sleep(0.05)
        pygame.mixer.music.unload()
        return True
    except Exception as exc:
        logger.warning("pygame playback failed: %s", exc)
        return False
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _run_async(coro) -> None:
    """
    Run an async coroutine safely whether or not an event loop is running.
    Handles the edge case where edge-tts is called from an async context.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside a running loop (e.g. FastAPI, async main) —
        # schedule on it and block from this thread.
        import concurrent.futures
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        fut.result(timeout=10)
    else:
        asyncio.run(coro)


def _speak_edge(text: str, lang: str) -> bool:
    try:
        import edge_tts

        voice = _EDGE_VOICES.get(lang, _EDGE_VOICES["en"])
        path  = tempfile.mktemp(suffix=".mp3")

        async def _generate():
            comm = edge_tts.Communicate(text, voice=voice)
            await comm.save(path)

        _run_async(_generate())

        if os.path.exists(path) and os.path.getsize(path) > 0:
            return _play_mp3(path)
        return False

    except ImportError:
        logger.debug("edge-tts not installed — pip install edge-tts")
        return False
    except RuntimeError as exc:
        # "cannot schedule new futures after interpreter shutdown" fires when
        # a daemon TTS thread outlives the main process.  Treat as silent
        # failure — the process is already exiting so audio doesn't matter.
        if "interpreter shutdown" in str(exc).lower():
            return False
        logger.warning("edge-tts RuntimeError: %s", exc)
        return False
    except Exception as exc:
        logger.warning("edge-tts failed: %s", exc)
        return False


def _speak_gtts(text: str, lang: str) -> bool:
    try:
        from gtts import gTTS

        path = tempfile.mktemp(suffix=".mp3")
        tld  = "co.in" if lang == "en" else "com"
        gTTS(text=text, lang=lang, tld=tld).save(path)
        return _play_mp3(path)

    except ImportError:
        logger.debug("gtts not installed — pip install gtts")
        return False
    except Exception as exc:
        logger.warning("gTTS failed: %s", exc)
        return False


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("Testing English (non-blocking)...")
    speak("Welcome to VisPay. Your voice guided payment assistant.", lang="en", force=True)
    wait_until_done()

    print("Testing Hindi (blocking)...")
    speak_and_wait("कृपया कैमरे की तरफ देखें।", lang="hi")

    print("Testing cooldown (should NOT speak)...")
    speak("Welcome to VisPay. Your voice guided payment assistant.")

    print("Testing interrupt=False (should skip if still playing)...")
    speak("This is a long sentence that takes a moment to finish.", force=True)
    speak("Low priority cue — skipped.", interrupt=False)

    wait_until_done()
    print("All tests done.")