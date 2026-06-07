"""
embed.py — Voice embedding utilities for VisPay

This is NOT a standalone script anymore.
It is a utility module imported by record.py and verify.py.

Provides:
  - extract_mfcc_embedding()  : lightweight baseline (for comparison/testing)
  - compare_embeddings()      : cosine similarity helper
  - embedding_exists()        : check before overwriting

The primary embedding method (ECAPA-TDNN) lives in record.py/verify.py
via SpeechBrain. This module handles storage and lightweight ops only.
"""

import os
import pickle
import numpy as np


SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "enrolled_voices")


# ── storage helpers ────────────────────────────────────────────────────

def embedding_exists(user_id: str) -> bool:
    """Return True if an embedding already exists for this user."""
    path = _embed_path(user_id)
    return os.path.exists(path)


def save_embedding(user_id: str, embedding: np.ndarray,
                   overwrite: bool = False) -> str:
    """
    Save embedding to disk.
    Raises FileExistsError if already enrolled and overwrite=False.
    Returns the saved path.
    """
    path = _embed_path(user_id)

    if os.path.exists(path) and not overwrite:
        raise FileExistsError(
            f"Embedding already exists for '{user_id}'. "
            f"Pass overwrite=True to replace it."
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(embedding, f)

    print(f"[embed] Saved: {path}")
    return path


def load_embedding(user_id: str) -> np.ndarray:
    """
    Load embedding for user_id.
    Raises FileNotFoundError if not enrolled.
    """
    path = _embed_path(user_id)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No embedding found for '{user_id}'. "
            f"Run record.py --user {user_id} first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def list_enrolled_users() -> list:
    """Return list of all enrolled user IDs."""
    if not os.path.exists(SAVE_DIR):
        return []
    return [
        d for d in os.listdir(SAVE_DIR)
        if os.path.isdir(os.path.join(SAVE_DIR, d))
        and os.path.exists(os.path.join(SAVE_DIR, d, "voice_embed.pkl"))
    ]


# ── comparison ─────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors. Range: -1 to 1."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ── lightweight MFCC baseline (for benchmarking only) ──────────────────

def extract_mfcc_embedding(audio_path: str,
                            n_mfcc: int = 40) -> np.ndarray:
    """
    MFCC mean embedding — kept as a baseline for comparison against ECAPA.
    NOT used in production verification. Used for ablation study.
    """
    import librosa
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)


# ── internal helpers ───────────────────────────────────────────────────

def _embed_path(user_id: str) -> str:
    return os.path.join(SAVE_DIR, user_id, "voice_embed.pkl")


# ── quick diagnostic ───────────────────────────────────────────────────

if __name__ == "__main__":
    users = list_enrolled_users()
    if users:
        print(f"Enrolled users ({len(users)}): {users}")
        for u in users:
            emb = load_embedding(u)
            print(f"  {u}: embedding shape={emb.shape}, "
                  f"norm={np.linalg.norm(emb):.4f}")
    else:
        print("No users enrolled yet. Run: python record.py --user <name>")