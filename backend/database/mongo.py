"""MongoDB integration and collection helpers."""

from datetime import datetime, timezone
from typing import Any, Optional

try:
    from bson import ObjectId
except ImportError:
    ObjectId = None

try:
    from pymongo import ASCENDING, MongoClient
    from pymongo.collection import Collection
    from pymongo.database import Database
except ImportError:
    ASCENDING = 1
    MongoClient = None
    Collection = Any
    Database = Any

from config import MONGO_URI, MONGO_DB_NAME
from utils.logger import get_logger

logger = get_logger("database.mongo")

_client: Optional[MongoClient] = None
_db: Optional[Database] = None


def _to_utc_now() -> datetime:
    return datetime.now(timezone.utc)


def init_mongo() -> Optional[Database]:
    global _client, _db
    if _db is not None:
        return _db

    if MongoClient is None:
        logger.warning("pymongo is not installed. Backend will run without persistent storage.")
        return None

    try:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        _client.admin.command("ping")
        _db = _client[MONGO_DB_NAME]
        _ensure_indexes(_db)
        logger.info(f"MongoDB connected: {MONGO_DB_NAME}")
        return _db
    except Exception as exc:
        logger.error(f"MongoDB connection failed: {exc}")
        _client = None
        _db = None
        return None


def get_db() -> Optional[Database]:
    return _db if _db is not None else init_mongo()


def _ensure_indexes(db: Database) -> None:
    db["users"].create_index([("email", ASCENDING)], unique=True)
    db["alerts"].create_index([("user_id", ASCENDING), ("timestamp", ASCENDING)])
    db["fog_predictions"].create_index([("timestamp", ASCENDING)])
    db["drowsiness_events"].create_index([("timestamp", ASCENDING)])
    db["emotion_events"].create_index([("timestamp", ASCENDING)])
    db["emotion_events"].create_index([("risk_level", ASCENDING), ("timestamp", ASCENDING)])
    db["otp_requests"].create_index([("email", ASCENDING)])
    db["otp_requests"].create_index([("expiry_time", ASCENDING)], expireAfterSeconds=0)


def _collection(name: str) -> Optional[Collection]:
    db = get_db()
    if db is None:
        return None
    return db[name]


def _serialize_id(record: Optional[dict]) -> Optional[dict]:
    if not record:
        return None
    if "_id" in record:
        record["id"] = str(record.pop("_id"))
    return record


def create_user(name: str, email: str, hashed_password: str) -> Optional[dict]:
    users = _collection("users")
    if users is None:
        return None

    payload = {
        "name": name,
        "email": email.lower().strip(),
        "hashed_password": hashed_password,
        "created_at": _to_utc_now(),
    }
    result = users.insert_one(payload)
    payload["_id"] = result.inserted_id
    return _serialize_id(payload)


def get_user_by_email(email: str) -> Optional[dict]:
    users = _collection("users")
    if users is None:
        return None
    record = users.find_one({"email": email.lower().strip()})
    return _serialize_id(record)


def get_user_by_id(user_id: str) -> Optional[dict]:
    users = _collection("users")
    if users is None:
        return None

    if ObjectId is None:
        return None

    try:
        oid = ObjectId(user_id)
    except Exception:
        return None

    record = users.find_one({"_id": oid})
    return _serialize_id(record)


def log_alert(user_id: str, alert_type: str, severity: str) -> Optional[str]:
    alerts = _collection("alerts")
    if alerts is None:
        return None

    payload = {
        "user_id": str(user_id),
        "alert_type": alert_type,
        "severity": severity,
        "timestamp": _to_utc_now(),
    }
    result = alerts.insert_one(payload)
    return str(result.inserted_id)


def log_fog_prediction(image_name: str, fog_probability: float) -> Optional[str]:
    fog_predictions = _collection("fog_predictions")
    if fog_predictions is None:
        return None

    payload = {
        "image_name": image_name,
        "fog_probability": float(fog_probability),
        "timestamp": _to_utc_now(),
    }
    result = fog_predictions.insert_one(payload)
    return str(result.inserted_id)


def log_drowsiness_event(ear_score: float, yawning_detected: bool) -> Optional[str]:
    drowsiness_events = _collection("drowsiness_events")
    if drowsiness_events is None:
        return None

    payload = {
        "ear_score": float(ear_score),
        "yawning_detected": bool(yawning_detected),
        "timestamp": _to_utc_now(),
    }
    result = drowsiness_events.insert_one(payload)
    return str(result.inserted_id)


def log_emotion_event(
    emotion: str,
    confidence: float,
    risk_level: str,
    risk_score: float,
    inference_ms: float,
) -> Optional[str]:
    emotion_events = _collection("emotion_events")
    if emotion_events is None:
        return None

    payload = {
        "emotion": str(emotion),
        "confidence": float(confidence),
        "risk_level": str(risk_level),
        "risk_score": float(risk_score),
        "inference_ms": float(inference_ms),
        "timestamp": _to_utc_now(),
    }
    result = emotion_events.insert_one(payload)
    return str(result.inserted_id)


def get_alerts(user_id: Optional[str] = None, limit: int = 100) -> list[dict[str, Any]]:
    alerts = _collection("alerts")
    if alerts is None:
        return []

    query: dict[str, Any] = {}
    if user_id:
        query["$or"] = [{"user_id": str(user_id)}, {"user_id": "system"}]

    rows = list(alerts.find(query).sort("timestamp", -1).limit(limit))
    for row in rows:
        _serialize_id(row)
        if isinstance(row.get("timestamp"), datetime):
            row["timestamp"] = row["timestamp"].isoformat()
    return rows


def get_drowsiness_events(limit: int = 100) -> list[dict[str, Any]]:
    events = _collection("drowsiness_events")
    if events is None:
        return []

    rows = list(events.find({}).sort("timestamp", -1).limit(limit))
    for row in rows:
        _serialize_id(row)
        if isinstance(row.get("timestamp"), datetime):
            row["timestamp"] = row["timestamp"].isoformat()
    return rows


def get_latest_emotion_event() -> Optional[dict[str, Any]]:
    emotion_events = _collection("emotion_events")
    if emotion_events is None:
        return None

    row = emotion_events.find_one({}, sort=[("timestamp", -1)])
    if row is None:
        return None

    _serialize_id(row)
    if isinstance(row.get("timestamp"), datetime):
        row["timestamp"] = row["timestamp"].isoformat()
    return row


# ── OTP Requests ─────────────────────────────────────────────────────

def create_otp_request(email: str, otp_code: str, expiry_minutes: int = 5) -> Optional[str]:
    """Store a new OTP request. Replaces any existing OTP for the email."""
    from datetime import timedelta
    col = _collection("otp_requests")
    if col is None:
        return None

    expiry_time = _to_utc_now() + timedelta(minutes=expiry_minutes)
    col.delete_many({"email": email.lower().strip()})
    result = col.insert_one({
        "email": email.lower().strip(),
        "otp_code": otp_code,
        "expiry_time": expiry_time,
        "created_at": _to_utc_now(),
    })
    return str(result.inserted_id)


def get_otp_request(email: str) -> Optional[dict]:
    """Retrieve the latest OTP request for an email."""
    col = _collection("otp_requests")
    if col is None:
        return None
    record = col.find_one({"email": email.lower().strip()})
    return _serialize_id(record)


def delete_otp_request(email: str) -> None:
    """Remove OTP requests for an email after use."""
    col = _collection("otp_requests")
    if col is not None:
        col.delete_many({"email": email.lower().strip()})


# ── Password Update ───────────────────────────────────────────────────

def update_user_password(email: str, hashed_password: str) -> bool:
    """Update the hashed_password for a user by email."""
    users = _collection("users")
    if users is None:
        return False
    result = users.update_one(
        {"email": email.lower().strip()},
        {"$set": {"hashed_password": hashed_password}},
    )
    return result.modified_count > 0
