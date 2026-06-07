"""
thresholds.py — Centralised threshold configuration for VisPay.

All similarity / confidence cut-offs live here so you can tune
the system in one place without touching pipeline code.

Cosine similarity ranges from -1 to 1; for FaceNet embeddings
practical scores sit in [0.3 – 0.99].  Higher = more similar.
"""

from dataclasses import dataclass, field


# ── Face verification ────────────────────────────────────────────────────────

@dataclass
class FaceThresholds:
    # Minimum cosine similarity to count a frame as a match
    match_similarity: float = 0.65

    # Fraction of VERIFY_FRAMES that must match to pass
    # e.g. 0.5 → majority vote (5 of 10 frames)
    majority_fraction: float = 0.5

    # How tightly a face must fill the frame during enrollment
    # (fraction of box dimension that keypoints must be inset)
    coverage_margin: float = 0.15

    # Minimum face bounding-box area (px²) to bother embedding
    min_face_area: int = 2_500      # 50×50 px


# ── Voice / speaker verification ────────────────────────────────────────────

@dataclass
class VoiceThresholds:
    # Cosine similarity between d-vector / x-vector embeddings
    match_similarity: float = 0.70

    # Minimum speech duration (seconds) to attempt verification
    min_speech_seconds: float = 1.5


# ── Confidence scoring (used by confidence.py) ───────────────────────────────

@dataclass
class ConfidenceWeights:
    # Weights must sum to 1.0
    face_weight:  float = 0.60
    voice_weight: float = 0.40

    # Fused score required to grant access
    access_threshold: float = 0.65

    def validate(self) -> None:
        total = round(self.face_weight + self.voice_weight, 6)
        if total != 1.0:
            raise ValueError(
                f"ConfidenceWeights must sum to 1.0, got {total}"
            )


# ── Sliding-window smoothing ─────────────────────────────────────────────────

@dataclass
class SmoothingConfig:
    # Number of consecutive frames to average before deciding
    window_size: int = 5

    # Exponential moving-average alpha (0 = lag, 1 = no smoothing)
    ema_alpha: float = 0.3


# ── Convenience singletons (import these directly) ───────────────────────────

FACE      = FaceThresholds()
VOICE     = VoiceThresholds()
FUSION    = ConfidenceWeights()
SMOOTHING = SmoothingConfig()

FUSION.validate()   # fail fast if someone edits weights incorrectly


# ── Helper ───────────────────────────────────────────────────────────────────

def summary() -> str:
    """Return a human-readable summary for logging / debugging."""
    lines = [
        "=== VisPay Threshold Configuration ===",
        f"  Face  match_similarity : {FACE.match_similarity}",
        f"  Face  majority_fraction: {FACE.majority_fraction}",
        f"  Face  coverage_margin  : {FACE.coverage_margin}",
        f"  Voice match_similarity : {VOICE.match_similarity}",
        f"  Voice min_speech_secs  : {VOICE.min_speech_seconds}",
        f"  Fusion face_weight     : {FUSION.face_weight}",
        f"  Fusion voice_weight    : {FUSION.voice_weight}",
        f"  Fusion access_threshold: {FUSION.access_threshold}",
        f"  Smoothing window_size  : {SMOOTHING.window_size}",
        f"  Smoothing ema_alpha    : {SMOOTHING.ema_alpha}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())