"""Analytics service for safety summary and scoring."""

from datetime import datetime, timezone

from config import (
    EMOTION_HIGH_RISK_WEIGHT,
    EMOTION_LOW_RISK_WEIGHT,
    EMOTION_MEDIUM_RISK_WEIGHT,
    EMOTION_WEIGHT,
)
from database.mongo import get_db


def _start_of_day_utc() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def generate_summary(user_id: str) -> dict:
    db = get_db()
    if db is None:
        return {
            "drowsiness_today": 0,
            "yawning_events": 0,
            "fog_alerts": 0,
            "stress_alerts": 0,
            "visibility_alerts": 0,
            "child_presence_alerts": 0,
            "emotion_events": 0,
            "emotion_high_risk_events": 0,
            "latest_emotion": "Neutral",
            "latest_emotion_confidence": 0.0,
            "emotion_risk_level": "Low",
            "safety_score": 100,
        }

    start_of_day = _start_of_day_utc()

    drowsiness_today = db["drowsiness_events"].count_documents({"timestamp": {"$gte": start_of_day}})
    yawning_events = db["drowsiness_events"].count_documents(
        {"timestamp": {"$gte": start_of_day}, "yawning_detected": True}
    )
    fog_alerts = db["alerts"].count_documents(
        {"timestamp": {"$gte": start_of_day}, "alert_type": "fog",
         "$or": [{"user_id": str(user_id)}, {"user_id": "system"}]}
    )
    stress_alerts = db["alerts"].count_documents(
        {
            "timestamp": {"$gte": start_of_day},
            "alert_type": "stress_high",
            "$or": [{"user_id": str(user_id)}, {"user_id": "system"}],
        }
    )
    visibility_alerts = db["alerts"].count_documents(
        {
            "timestamp": {"$gte": start_of_day},
            "alert_type": {"$in": ["visibility_fog", "visibility_low-light"]},
            "$or": [{"user_id": str(user_id)}, {"user_id": "system"}],
        }
    )
    child_presence_alerts = db["alerts"].count_documents(
        {
            "timestamp": {"$gte": start_of_day},
            "alert_type": "child_presence",
            "$or": [{"user_id": str(user_id)}, {"user_id": "system"}],
        }
    )

    fog_cursor = db["fog_predictions"].find({"timestamp": {"$gte": start_of_day}}, {"fog_probability": 1})
    fog_probs = [float(row.get("fog_probability", 0.0)) for row in fog_cursor]
    avg_fog_prob = sum(fog_probs) / len(fog_probs) if fog_probs else 0.0

    emotion_cursor = list(
        db["emotion_events"].find(
            {"timestamp": {"$gte": start_of_day}},
            {"emotion": 1, "confidence": 1, "risk_level": 1, "risk_score": 1},
        )
    )
    emotion_events = len(emotion_cursor)
    emotion_high_risk_events = sum(1 for row in emotion_cursor if row.get("risk_level") == "High")
    emotion_medium_risk_events = sum(1 for row in emotion_cursor if row.get("risk_level") == "Medium")
    emotion_low_risk_events = sum(1 for row in emotion_cursor if row.get("risk_level") == "Low")
    avg_emotion_score = (
        sum(float(row.get("risk_score", 0.0)) for row in emotion_cursor) / emotion_events if emotion_events else 0.0
    )
    latest_emotion_event = max(emotion_cursor, key=lambda row: row.get("_id"), default=None)

    emotion_risk_score = (
        (emotion_high_risk_events * EMOTION_HIGH_RISK_WEIGHT)
        + (emotion_medium_risk_events * EMOTION_MEDIUM_RISK_WEIGHT)
        + (emotion_low_risk_events * EMOTION_LOW_RISK_WEIGHT)
        + (avg_emotion_score * EMOTION_WEIGHT)
    )

    risk_score = (
        (drowsiness_today * 5)
        + (yawning_events * 3)
        + (avg_fog_prob * 10)
        + (stress_alerts * 4)
        + (visibility_alerts * 3)
        + (child_presence_alerts * 8)
        + emotion_risk_score
    )
    safety_score = int(round(_clamp(100 - risk_score, 0, 100)))

    return {
        "drowsiness_today": int(drowsiness_today),
        "yawning_events": int(yawning_events),
        "fog_alerts": int(fog_alerts),
        "stress_alerts": int(stress_alerts),
        "visibility_alerts": int(visibility_alerts),
        "child_presence_alerts": int(child_presence_alerts),
        "emotion_events": int(emotion_events),
        "emotion_high_risk_events": int(emotion_high_risk_events),
        "latest_emotion": (latest_emotion_event or {}).get("emotion", "Neutral"),
        "latest_emotion_confidence": float((latest_emotion_event or {}).get("confidence", 0.0)),
        "emotion_risk_level": (latest_emotion_event or {}).get("risk_level", "Low"),
        "emotion_risk_score": round(emotion_risk_score, 2),
        "safety_score": safety_score,
    }
