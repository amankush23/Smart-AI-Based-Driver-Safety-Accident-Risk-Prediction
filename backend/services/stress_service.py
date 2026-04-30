"""Voice stress detection service with safe fallbacks for dashboard integration."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from database.mongo import log_alert
from services.audio_alert_service import trigger_alert
from utils.logger import get_logger

logger = get_logger("stress_service")

try:
    import librosa
except Exception:
    librosa = None

try:
    from scipy.io import wavfile
except Exception:
    wavfile = None

SR = 22050
N_MFCC = 13

LABELS = {0: "Normal", 1: "Mild Stress", 2: "High Stress"}
SCORES = {0: 8.0, 1: 45.0, 2: 80.0}
STATE = {
    "active": False,
    "level": "Normal",
    "label": 0,
    "confidence": 0.0,
    "score": 0.0,
    "source": "idle",
    "timestamp": 0.0,
}

_was_high_stress = False


@dataclass
class _StressModel:
    scaler: StandardScaler
    model: RandomForestClassifier


_model: _StressModel | None = None


def _extract_features(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if audio.size == 0:
        return np.zeros(N_MFCC * 2 + 6, dtype=np.float32)

    if librosa is None:
        # Fallback features without librosa (fixed 32 dims).
        zc = np.mean(np.abs(np.diff(np.sign(audio)))) if audio.size > 1 else 0.0
        rms = np.sqrt(np.mean(np.square(audio))) if audio.size else 0.0
        base = np.array([
            float(np.mean(audio)),
            float(np.std(audio)),
            float(np.max(audio)),
            float(np.min(audio)),
            float(np.mean(np.abs(audio))),
            float(rms),
            float(zc),
            float(np.percentile(audio, 75) - np.percentile(audio, 25)),
        ], dtype=np.float32)
        tiled = np.tile(base, 4)
        return tiled[: (N_MFCC * 2 + 6)].astype(np.float32)

    feats: list[float] = []
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    feats.extend(np.mean(mfcc, axis=1).tolist())
    feats.extend(np.std(mfcc, axis=1).tolist())

    try:
        f0, voiced_flag, _ = librosa.pyin(audio, fmin=50, fmax=500, sr=sr)
        pitched = f0[voiced_flag] if voiced_flag is not None else np.array([0.0])
        feats += [
            float(np.nanmean(pitched)) if pitched.size else 0.0,
            float(np.nanstd(pitched)) if pitched.size else 0.0,
        ]
    except Exception:
        feats += [0.0, 0.0]

    rms = librosa.feature.rms(y=audio)
    zcr = librosa.feature.zero_crossing_rate(audio)
    feats += [float(np.mean(rms)), float(np.std(rms)), float(np.mean(zcr)), float(np.std(zcr))]
    return np.array(feats, dtype=np.float32)


def _build_demo_model() -> _StressModel:
    np.random.seed(42)
    n = 500
    nf = N_MFCC * 2 + 6
    X = np.vstack(
        [
            np.random.randn(n, nf) * 0.40,
            np.random.randn(n, nf) * 0.70 + 0.50,
            np.random.randn(n, nf) * 1.10 + 1.20,
        ]
    )
    y = np.array([0] * n + [1] * n + [2] * n)

    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    scaler.fit(Xt)

    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(scaler.transform(Xt), yt)
    acc = model.score(scaler.transform(Xv), yv)
    logger.info("Stress detector demo model ready (validation accuracy=%.0f%%)", acc * 100)
    return _StressModel(scaler=scaler, model=model)


def _ensure_model() -> _StressModel:
    global _model
    if _model is None:
        _model = _build_demo_model()
    return _model


def _update_state(level_id: int, confidence: float, source: str) -> dict[str, Any]:
    global _was_high_stress
    now = time.time()

    STATE.update(
        {
            "active": True,
            "level": LABELS[level_id],
            "label": int(level_id),
            "confidence": round(float(confidence), 2),
            "score": round(float(SCORES[level_id]), 1),
            "source": source,
            "timestamp": now,
        }
    )

    is_high_stress = (level_id == 2)
    if is_high_stress and not _was_high_stress:
        log_alert(user_id="system", alert_type="stress_high", severity="high")
        trigger_alert("stress", cooldown_seconds=4.0)

    _was_high_stress = is_high_stress

    return dict(STATE)


def predict_from_audio(audio: np.ndarray, sr: int = SR, source: str = "audio") -> dict[str, Any]:
    model_bundle = _ensure_model()
    feats = _extract_features(audio, sr).reshape(1, -1)
    scaled = model_bundle.scaler.transform(feats)
    label = int(model_bundle.model.predict(scaled)[0])
    probs = model_bundle.model.predict_proba(scaled)[0]
    confidence = float(np.max(probs))
    return _update_state(label, confidence, source)


def predict_from_bytes(contents: bytes, filename: str = "audio.wav") -> dict[str, Any]:
    if librosa is not None:
        audio, sr = librosa.load(io.BytesIO(contents), sr=SR, mono=True)
        return predict_from_audio(audio, sr, source=f"upload:{filename}")

    if wavfile is not None:
        sr, audio = wavfile.read(io.BytesIO(contents))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if np.issubdtype(audio.dtype, np.integer):
            max_val = max(1, np.iinfo(audio.dtype).max)
            audio = audio.astype(np.float32) / float(max_val)
        return predict_from_audio(audio.astype(np.float32), int(sr), source=f"upload:{filename}")

    logger.warning("librosa/scipy not available for audio decoding; returning neutral stress state")
    return _update_state(0, 0.0, source="fallback")


def estimate_from_context(drowsiness_state: dict[str, Any] | None) -> dict[str, Any]:
    """Estimate stress score from driving behavior when audio is unavailable."""
    if not drowsiness_state or not drowsiness_state.get("active"):
        return _update_state(0, 0.55, source="context")

    if drowsiness_state.get("drowsy"):
        return _update_state(2, 0.86, source="context")
    if drowsiness_state.get("yawning"):
        return _update_state(1, 0.74, source="context")

    ear = float(drowsiness_state.get("ear", 0.30) or 0.30)
    if ear < 0.23:
        return _update_state(1, 0.67, source="context")
    return _update_state(0, 0.62, source="context")


def get_state() -> dict[str, Any]:
    return dict(STATE)