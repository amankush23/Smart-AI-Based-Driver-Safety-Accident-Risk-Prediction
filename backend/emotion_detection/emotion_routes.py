"""Emotion detection API routes."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from database.mongo import get_latest_emotion_event
from emotion_detection.emotion_predictor import (
    decode_base64_frame,
    decode_upload_bytes,
    predict_from_frame,
    prediction_to_dict,
)

router = APIRouter(prefix="/emotion-detection", tags=["emotion-detection"])


def _fallback_emotion_response(reason: str) -> dict:
    return {
        "emotion": "Neutral",
        "confidence": 0.0,
        "risk_level": "Low",
        "risk_score": 0.0,
        "driver_risk_score": 0,
        "inference_ms": 0.0,
        "icon": ":|",
        "alert": False,
        "model_unavailable": True,
        "reason": reason,
    }


class EmotionFrameRequest(BaseModel):
    frame: str


@router.post("/predict")
async def predict_emotion_from_upload(file: Optional[UploadFile] = File(default=None)) -> dict:
    if file is None:
        raise HTTPException(status_code=400, detail="A frame image file is required.")

    try:
        frame = decode_upload_bytes(await file.read())
        prediction = predict_from_frame(frame)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        message = str(exc)
        if "TensorFlow is required" in message:
            return _fallback_emotion_response(message)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return prediction_to_dict(prediction)


@router.post("/predict-base64")
def predict_emotion_from_base64(payload: EmotionFrameRequest) -> dict:
    try:
        frame = decode_base64_frame(payload.frame)
        prediction = predict_from_frame(frame)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        message = str(exc)
        if "TensorFlow is required" in message:
            return _fallback_emotion_response(message)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return prediction_to_dict(prediction)


@router.get("/latest")
def get_latest_emotion_prediction() -> dict:
    latest = get_latest_emotion_event()
    if latest is None:
        return {
            "emotion": "Neutral",
            "confidence": 0.0,
            "risk_level": "Low",
            "risk_score": 0.0,
            "driver_risk_score": 0,
            "inference_ms": 0.0,
            "icon": ":|",
            "alert": False,
        }
    latest["alert"] = latest.get("risk_level") == "High"
    latest.setdefault("icon", ":|")
    return latest
