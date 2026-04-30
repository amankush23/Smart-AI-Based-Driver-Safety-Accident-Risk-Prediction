"""Kid safety detection service using OpenCV DNN age estimation."""

from __future__ import annotations

import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from config import MODELS_DIR
from database.mongo import log_alert
from utils.logger import get_logger

logger = get_logger("kid_safety_service")

ALONE_KID_TIME_THRESHOLD = float(os.getenv("KID_SAFETY_ALONE_TIME_THRESHOLD", 2.0))
CONF_THRESHOLD = float(os.getenv("KID_SAFETY_FACE_CONF_THRESHOLD", 0.6))
DEBUG_BOXES = os.getenv("KID_SAFETY_DEBUG_BOXES", "false").lower() == "true"
MODEL_LOAD_RETRY_SECONDS = float(os.getenv("KID_SAFETY_MODEL_RETRY_SECONDS", 30.0))

FACE_PROTO = "opencv_face_detector.pbtxt"
FACE_MODEL = "opencv_face_detector_uint8.pb"
AGE_PROTO = "age_deploy.prototxt"
AGE_MODEL = "age_net.caffemodel"

AGE_BUCKETS = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)",
]
KID_BUCKETS = {"(0-2)", "(4-6)", "(8-12)"}
ADULT_BUCKETS = {"(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"}

MODEL_URLS = {
    FACE_PROTO: [
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector.pbtxt",
    ],
    FACE_MODEL: [
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector_uint8.pb",
    ],
    AGE_PROTO: [
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt",
    ],
    AGE_MODEL: [
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_net.caffemodel",
        "https://github.com/GilLevi/AgeGenderDeepLearning/raw/refs/heads/master/models/age_net.caffemodel",
    ],
}

_MODEL_DIR = MODELS_DIR / "kid_safety"
_lock = threading.Lock()
_face_net = None
_age_net = None
_models_ready = False
_last_model_load_attempt_ts = 0.0
_alone_started_at: Optional[float] = None
_last_danger_alert_ts = 0.0

_state: dict[str, Any] = {
    "active": False,
    "kid_detected": False,
    "adult_present": False,
    "status": "NO_FACE",
    "risk": 0.0,
    "message": "No occupant detected",
    "alone_seconds": 0.0,
    "boxes": [],
    "timestamp": 0.0,
}


def ensure_model_files(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    for filename, urls in MODEL_URLS.items():
        full_path = base_dir / filename
        if full_path.is_file():
            continue
        downloaded = False
        for url in urls:
            try:
                logger.info("Downloading missing kid safety model file: %s", filename)
                urllib.request.urlretrieve(url, full_path)
                downloaded = True
                break
            except Exception:
                continue
        if not downloaded:
            raise RuntimeError(f"Failed to download model file: {filename}")


def load_nets(base_dir: Path):
    face_net = cv2.dnn.readNet(
        str(base_dir / FACE_MODEL),
        str(base_dir / FACE_PROTO),
    )
    age_net = cv2.dnn.readNetFromCaffe(
        str(base_dir / AGE_PROTO),
        str(base_dir / AGE_MODEL),
    )
    return face_net, age_net


def load_model() -> None:
    """Load the OpenCV DNN age/face models once at startup."""
    global _face_net, _age_net, _models_ready, _last_model_load_attempt_ts
    if _models_ready:
        return
    now = time.time()
    if _last_model_load_attempt_ts and now - _last_model_load_attempt_ts < MODEL_LOAD_RETRY_SECONDS:
        return
    _last_model_load_attempt_ts = now
    try:
        ensure_model_files(_MODEL_DIR)
        _face_net, _age_net = load_nets(_MODEL_DIR)
        _models_ready = True
        logger.info("Kid safety models loaded from %s", _MODEL_DIR)
    except Exception as exc:
        logger.error("Failed to load kid safety models: %s", exc)
        _face_net = None
        _age_net = None
        _models_ready = False


def _ensure_models() -> bool:
    if not _models_ready or _face_net is None or _age_net is None:
        load_model()
    return bool(_models_ready and _face_net is not None and _age_net is not None)


def _decode_frame(image_bytes: bytes) -> np.ndarray | None:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def detect_faces(frame: np.ndarray, face_net) -> list[tuple[int, int, int, int, float]]:
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
        swapRB=False,
        crop=False,
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    boxes: list[tuple[int, int, int, int, float]] = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < CONF_THRESHOLD:
            continue
        x1 = max(0, int(detections[0, 0, i, 3] * w))
        y1 = max(0, int(detections[0, 0, i, 4] * h))
        x2 = min(w - 1, int(detections[0, 0, i, 5] * w))
        y2 = min(h - 1, int(detections[0, 0, i, 6] * h))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2, confidence))
    return boxes


def classify_age(face_img: np.ndarray, age_net) -> str:
    blob = cv2.dnn.blobFromImage(
        face_img,
        scalefactor=1.0,
        size=(227, 227),
        mean=(78.4263377603, 87.7689143744, 114.895847746),
        swapRB=False,
    )
    age_net.setInput(blob)
    preds = age_net.forward()[0]
    age_idx = int(np.argmax(preds))
    return AGE_BUCKETS[age_idx]


def _status_payload(
    *,
    kid_detected: bool,
    adult_present: bool,
    status: str,
    risk: float,
    message: str,
    alone_seconds: float,
    boxes: list[dict[str, Any]],
    active: bool = True,
) -> dict[str, Any]:
    return {
        "active": active,
        "kid_detected": kid_detected,
        "adult_present": adult_present,
        "status": status,
        "risk": round(float(risk), 1),
        "message": message,
        "alone_seconds": round(float(alone_seconds), 2),
        "boxes": boxes,
        "timestamp": time.time(),
    }


def _update_from_frame(frame: np.ndarray | None, user_id: str) -> dict[str, Any]:
    global _alone_started_at, _last_danger_alert_ts

    if frame is None:
        return _status_payload(
            kid_detected=False,
            adult_present=False,
            status="NO_FACE",
            risk=0.0,
            message="No occupant detected",
            alone_seconds=0.0,
            boxes=[],
            active=False,
        )

    if not _ensure_models():
        return _status_payload(
            kid_detected=False,
            adult_present=False,
            status="NO_FACE",
            risk=0.0,
            message="Kid safety model unavailable",
            alone_seconds=0.0,
            boxes=[],
            active=False,
        )

    assert _face_net is not None
    assert _age_net is not None

    faces = detect_faces(frame, _face_net)
    if not faces:
        _alone_started_at = None
        return _status_payload(
            kid_detected=False,
            adult_present=False,
            status="NO_FACE",
            risk=0.0,
            message="No occupant detected",
            alone_seconds=0.0,
            boxes=[],
        )

    kid_detected = False
    adult_present = False
    debug_boxes: list[dict[str, Any]] = []

    for x1, y1, x2, y2, confidence in faces:
        face_crop = frame[max(0, y1 - 10):min(frame.shape[0], y2 + 10), max(0, x1 - 10):min(frame.shape[1], x2 + 10)]
        if face_crop.size == 0:
            continue

        age_bucket = classify_age(face_crop, _age_net)
        if age_bucket in KID_BUCKETS:
            kid_detected = True
            category = "kid"
            color = (0, 200, 255)
        else:
            if age_bucket in ADULT_BUCKETS or age_bucket == "(15-20)":
                adult_present = True
            category = "adult"
            color = (0, 255, 0)

        debug_boxes.append(
            {
                "left": x1,
                "top": y1,
                "right": x2,
                "bottom": y2,
                "confidence": round(confidence, 3),
                "age_bucket": age_bucket,
                "category": category,
                "color": color,
            }
        )

    now = time.time()
    if kid_detected and adult_present:
        _alone_started_at = None
        payload = _status_payload(
            kid_detected=True,
            adult_present=True,
            status="SAFE",
            risk=5.0,
            message="Adult present with child",
            alone_seconds=0.0,
            boxes=debug_boxes,
        )
    elif kid_detected and not adult_present:
        if _alone_started_at is None:
            _alone_started_at = now
        alone_seconds = max(0.0, now - _alone_started_at)
        if alone_seconds < ALONE_KID_TIME_THRESHOLD:
            payload = _status_payload(
                kid_detected=True,
                adult_present=False,
                status="WARNING",
                risk=40.0,
                message="Child detected (monitoring)",
                alone_seconds=alone_seconds,
                boxes=debug_boxes,
            )
        else:
            payload = _status_payload(
                kid_detected=True,
                adult_present=False,
                status="DANGER",
                risk=95.0,
                message="Child alone in car",
                alone_seconds=alone_seconds,
                boxes=debug_boxes,
            )
            if now - _last_danger_alert_ts >= 10:
                log_alert(user_id=user_id, alert_type="kid_safety", severity="critical")
                _last_danger_alert_ts = now
    else:
        _alone_started_at = None
        payload = _status_payload(
            kid_detected=False,
            adult_present=adult_present,
            status="NORMAL",
            risk=5.0,
            message="No risk",
            alone_seconds=0.0,
            boxes=debug_boxes,
        )

    if DEBUG_BOXES:
        payload["debug_boxes"] = debug_boxes

    return payload


def predict(image_bytes: bytes, user_id: str = "system", image_name: str = "camera_frame.jpg") -> dict[str, Any]:
    frame = _decode_frame(image_bytes)
    payload = _update_from_frame(frame, user_id)

    with _lock:
        _state.update(payload)
        _state["active"] = bool(payload.get("active", True))
        _state["timestamp"] = payload["timestamp"]
    return get_state()


def get_state() -> dict[str, Any]:
    with _lock:
        return dict(_state)
