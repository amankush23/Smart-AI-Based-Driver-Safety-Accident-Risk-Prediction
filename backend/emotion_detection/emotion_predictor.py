"""Emotion prediction pipeline for uploaded webcam frames."""

from __future__ import annotations

import base64
import binascii
import sys
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any

import cv2
import numpy as np

from config import EMOTION_INPUT_SIZE
from database.mongo import log_alert, log_emotion_event
from emotion_detection.emotion_model_loader import EmotionAssets, EmotionModelLoader
from utils.logger import get_logger

logger = get_logger("emotion.predictor")

_PREDICTION_LOCK = Lock()
_ALERT_LOCK = Lock()
_LAST_HIGH_RISK_ALERT_TS = 0.0

EMOTION_LABELS = {
    "fearful": "Fear",
    "surprised": "Surprised",
    "disgusted": "Disgusted",
    "happy": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",
    "angry": "Angry",
    "stress": "Stress",
    "stressed": "Stress",
}

RISK_LEVELS = {
    "angry": "High",
    "stress": "High",
    "stressed": "High",
    "sad": "Medium",
    "fear": "Medium",
    "fearful": "Medium",
    "disgusted": "Medium",
    "surprised": "Medium",
    "neutral": "Low",
    "happy": "Low",
}

RISK_SCORES = {
    "High": 24.0,
    "Medium": 12.0,
    "Low": 4.0,
}

EMOTION_ICONS = {
    "Angry": "!!",
    "Stress": "##",
    "Sad": ":(",
    "Fear": ":o",
    "Neutral": ":|",
    "Happy": ":)",
    "Disgusted": "xx",
    "Surprised": ":O",
}


@dataclass
class EmotionPrediction:
    emotion: str
    confidence: float
    risk_level: str
    risk_score: float
    driver_risk_score: int
    inference_ms: float
    icon: str


def _canonical_label(label: str) -> str:
    normalized = str(label).strip().lower()
    if normalized == "fearful":
        normalized = "fear"
    if normalized in EMOTION_LABELS:
        return EMOTION_LABELS[normalized]
    return normalized.capitalize() if normalized else "Unknown"


def _risk_level(label: str) -> str:
    normalized = str(label).strip().lower()
    return RISK_LEVELS.get(normalized, RISK_LEVELS.get(normalized.replace("fearful", "fear"), "Low"))


def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Unable to decode frame image.")
    return frame


def decode_base64_frame(frame_data: str) -> np.ndarray:
    payload = frame_data.split(",", 1)[-1]
    try:
        return _decode_image_bytes(base64.b64decode(payload))
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid base64 frame payload.") from exc


def decode_upload_bytes(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("Frame payload is empty.")
    return _decode_image_bytes(image_bytes)


def preprocess_frame(frame_bgr: np.ndarray, target_size: tuple[int, int] = (EMOTION_INPUT_SIZE, EMOTION_INPUT_SIZE)) -> np.ndarray:
    frame = extract_face_region(frame_bgr)
    resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    if sys.version_info >= (3, 13):
        processed = (rgb_frame / 127.5) - 1.0
    else:
        try:
            from tensorflow.keras.applications.efficientnet import preprocess_input

            processed = preprocess_input(rgb_frame)
        except ImportError:
            processed = (rgb_frame / 127.5) - 1.0
    return np.expand_dims(processed, axis=0)


def extract_face_region(frame_bgr: np.ndarray) -> np.ndarray:
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        return frame_bgr

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return frame_bgr

    x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
    return frame_bgr[y : y + height, x : x + width]


def _predict_probabilities(assets: EmotionAssets, batch: np.ndarray) -> np.ndarray:
    with _PREDICTION_LOCK:
        probabilities = assets.model.predict(batch, verbose=0)[0]
    return np.ravel(probabilities)


def _estimate_driver_risk(risk_score: float, confidence: float) -> int:
    return max(0, min(100, int(round(risk_score * (0.55 + confidence)))))


def _maybe_emit_high_risk_alert(emotion: str, risk_level: str) -> None:
    global _LAST_HIGH_RISK_ALERT_TS
    if risk_level != "High":
        return

    now = time.monotonic()
    with _ALERT_LOCK:
        if now - _LAST_HIGH_RISK_ALERT_TS < 10.0:
            return
        _LAST_HIGH_RISK_ALERT_TS = now
    log_alert(user_id="system", alert_type=f"emotion_{emotion.lower()}", severity="high")


def predict_from_frame(frame_bgr: np.ndarray, loader: EmotionModelLoader | None = None) -> EmotionPrediction:
    start = time.perf_counter()
    assets = (loader or EmotionModelLoader.instance()).get_assets()
    batch = preprocess_frame(frame_bgr)
    probabilities = _predict_probabilities(assets, batch)

    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])
    raw_label = assets.class_names[index] if index < len(assets.class_names) else str(index)
    emotion = _canonical_label(raw_label)
    risk_level = _risk_level(raw_label)
    risk_score = float(RISK_SCORES[risk_level] * max(confidence, 0.35))
    driver_risk_score = _estimate_driver_risk(risk_score, confidence)
    inference_ms = round((time.perf_counter() - start) * 1000, 2)

    log_emotion_event(
        emotion=emotion,
        confidence=confidence,
        risk_level=risk_level,
        risk_score=risk_score,
        inference_ms=inference_ms,
    )
    _maybe_emit_high_risk_alert(emotion, risk_level)

    return EmotionPrediction(
        emotion=emotion,
        confidence=confidence,
        risk_level=risk_level,
        risk_score=risk_score,
        driver_risk_score=driver_risk_score,
        inference_ms=inference_ms,
        icon=EMOTION_ICONS.get(emotion, "::"),
    )


def prediction_to_dict(prediction: EmotionPrediction) -> dict[str, Any]:
    return {
        "emotion": prediction.emotion,
        "confidence": round(prediction.confidence, 4),
        "risk_level": prediction.risk_level,
        "risk_score": round(prediction.risk_score, 2),
        "driver_risk_score": prediction.driver_risk_score,
        "inference_ms": prediction.inference_ms,
        "icon": prediction.icon,
        "alert": prediction.risk_level == "High",
    }
