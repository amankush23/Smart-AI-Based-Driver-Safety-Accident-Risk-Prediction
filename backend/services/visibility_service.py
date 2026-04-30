"""Visibility and child-presence detection service based on frame analysis."""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import cv2
import numpy as np

from database.mongo import log_alert
from utils.logger import get_logger

logger = get_logger("visibility_service")

BRIGHT_THRESH = 40
CONTRAST_THRESH = 25
BLUR_THRESH = 80
MOTION_THRESH = 8.0
MOTION_MIN_AREA = 1200

VIS_LABELS = {0: "Clear", 1: "Low-Light", 2: "Fog", 3: "Blurry"}
VIS_SCORES = {0: 8.0, 1: 55.0, 2: 75.0, 3: 45.0}

_last_visibility_alert_ts = 0.0
_last_child_alert_ts = 0.0
_prev_gray: np.ndarray | None = None
_motion_buf: deque[int] = deque(maxlen=30)

STATE = {
    "active": False,
    "visibility": {
        "condition": "Unknown",
        "cid": 0,
        "brightness": 0.0,
        "contrast": 0.0,
        "blur_var": 0.0,
        "score": 0.0,
    },
    "child_presence": {
        "engine_on": True,
        "motion": False,
        "alert": False,
        "recent_pct": 0.0,
        "score": 0.0,
    },
    "timestamp": 0.0,
}


def set_engine(on: bool) -> dict[str, Any]:
    global _prev_gray
    STATE["child_presence"]["engine_on"] = bool(on)
    if on:
        _prev_gray = None
    return get_state()


def _decode_frame(image_bytes: bytes) -> np.ndarray | None:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    if arr.size == 0:
        return None
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def _analyze_visibility(frame: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if brightness < BRIGHT_THRESH:
        cid = 1
    elif contrast < CONTRAST_THRESH:
        cid = 2
    elif blur_var < BLUR_THRESH:
        cid = 3
    else:
        cid = 0

    return {
        "condition": VIS_LABELS[cid],
        "cid": cid,
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "blur_var": round(blur_var, 1),
        "score": round(VIS_SCORES[cid], 1),
    }


def _detect_child_presence(frame: np.ndarray) -> dict[str, Any]:
    global _prev_gray

    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    motion = False

    if _prev_gray is not None:
        diff = cv2.absdiff(_prev_gray, gray)
        motion = float(np.mean(diff)) > MOTION_THRESH

        thresh = cv2.dilate(cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1], None, iterations=2)
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not motion:
            for c in cnts:
                if cv2.contourArea(c) > MOTION_MIN_AREA:
                    motion = True
                    break

    _prev_gray = gray.copy()
    _motion_buf.append(1 if motion else 0)

    engine_on = bool(STATE["child_presence"].get("engine_on", True))
    alert = (not engine_on) and motion

    return {
        "engine_on": engine_on,
        "motion": motion,
        "alert": alert,
        "recent_pct": round(sum(_motion_buf) / max(1, len(_motion_buf)), 2),
        "score": 95.0 if alert else 0.0,
    }


def predict(image_bytes: bytes, user_id: str = "system", image_name: str = "frame.jpg") -> dict[str, Any]:
    global _last_visibility_alert_ts, _last_child_alert_ts

    frame = _decode_frame(image_bytes)
    if frame is None:
        return get_state()

    vis = _analyze_visibility(frame)
    child = _detect_child_presence(frame)

    STATE["active"] = True
    STATE["visibility"] = vis
    STATE["child_presence"] = child
    STATE["timestamp"] = time.time()

    now = time.time()
    if vis["condition"] in {"Fog", "Low-Light"} and now - _last_visibility_alert_ts >= 20:
        severity = "high" if vis["condition"] == "Fog" else "medium"
        log_alert(user_id=user_id, alert_type=f"visibility_{vis['condition'].lower()}", severity=severity)
        _last_visibility_alert_ts = now

    if child["alert"] and now - _last_child_alert_ts >= 10:
        log_alert(user_id=user_id, alert_type="child_presence", severity="critical")
        _last_child_alert_ts = now

    return get_state()


def get_state() -> dict[str, Any]:
    return {
        "active": bool(STATE["active"]),
        "visibility": dict(STATE["visibility"]),
        "child_presence": dict(STATE["child_presence"]),
        "timestamp": STATE["timestamp"],
    }