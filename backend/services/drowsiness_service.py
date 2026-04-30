"""Drowsiness and yawn detection service (webcam background thread)."""

from __future__ import annotations

import math
import os
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlopen

import numpy as np

from config import (
    BOX_SMOOTH_ALPHA,
    EYE_AR_CONSEC_FRAMES,
    EYE_AR_THRESH,
    HEAD_POSE_ALERT_SECONDS,
    HEAD_POSE_PITCH_THRESH,
    HEAD_POSE_RETURN_RATIO,
    HEAD_POSE_YAW_THRESH,
    METRIC_SMOOTH_ALPHA,
    MODELS_DIR,
    YAWN_CLOSE_RATIO,
    YAWN_MIN_CONSEC_FRAMES,
    YAWN_MIN_DURATION_SECONDS,
    YAWN_OPEN_RATIO,
    YAWN_THRESH,
    YAWN_CONSEC_FRAMES,
)
from database.mongo import log_alert, log_drowsiness_event
from services.audio_alert_service import start_alert_loop, stop_alert
from utils.logger import get_logger

logger = get_logger("drowsiness_service")

_FACE_LANDMARKER_MODEL = MODELS_DIR / "face_landmarker.task"
_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# ── Thread-safe shared state ─────────────────────────────────────────
_lock = threading.Lock()
_state: dict = {
    "active": False,
    "backend": None,
    "face_detected": False,
    "drowsy": False,
    "yawning": False,
    "ear": None,
    "counter": 0,
    "head_pose": {
        "direction": "forward",
        "yaw": 0.0,
        "pitch": 0.0,
        "off_center": False,
        "alert": False,
        "seconds": 0.0,
    },
    "mouth": {
        "ratio": None,
        "smoothed_ratio": None,
        "open_seconds": 0.0,
    },
    "boxes": {
        "face": None,
        "eyes": None,
        "mouth": None,
    },
    "alert_message": None,
    "timestamp": 0.0,
}
_latest_frame_jpeg: Optional[bytes] = None
_running = False
_thread: Optional[threading.Thread] = None
_last_event_log_ts = 0.0
_mediapipe_backend_supported: Optional[bool] = None

_FACE_LANDMARK_INDICES = tuple(range(468))
_LEFT_EYE_INDICES = (33, 133, 160, 159, 158, 157, 173, 144, 145, 153)
_RIGHT_EYE_INDICES = (362, 263, 387, 386, 385, 384, 398, 373, 374, 380)
_MOUTH_INDICES = (61, 291, 13, 14, 78, 308, 80, 81, 82, 87, 88, 95, 178, 317, 318, 402, 405)

_BOX_COLORS = {
    "normal": "#00c853",
    "warning": "#ffb300",
    "risk": "#ff3b30",
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _smooth_metric(previous: Optional[float], current: Optional[float], alpha: float = METRIC_SMOOTH_ALPHA) -> Optional[float]:
    if current is None:
        return None
    if previous is None:
        return current
    return ((1.0 - alpha) * previous) + (alpha * current)


def _smooth_box(previous: Optional[dict[str, float]], current: Optional[dict[str, float]], alpha: float = BOX_SMOOTH_ALPHA) -> Optional[dict[str, float]]:
    if current is None:
        return None
    if previous is None:
        return dict(current)
    return {
        key: ((1.0 - alpha) * float(previous.get(key, current[key])) + (alpha * float(value)))
        for key, value in current.items()
    }


def _box_payload(box: Optional[dict[str, float]], status: str = "normal") -> Optional[dict[str, Any]]:
    if box is None:
        return None
    payload = dict(box)
    payload["status"] = status
    payload["color"] = _BOX_COLORS.get(status, _BOX_COLORS["normal"])
    return payload


def _points_to_box(points: list[tuple[float, float]], padding: float = 0.04) -> Optional[dict[str, float]]:
    if not points:
        return None

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left = _clamp(min(xs) - padding, 0.0, 1.0)
    top = _clamp(min(ys) - padding, 0.0, 1.0)
    right = _clamp(max(xs) + padding, 0.0, 1.0)
    bottom = _clamp(max(ys) + padding, 0.0, 1.0)
    if right <= left or bottom <= top:
        return None

    return {
        "left": float(left),
        "top": float(top),
        "right": float(right),
        "bottom": float(bottom),
        "width": float(right - left),
        "height": float(bottom - top),
        "center_x": float((left + right) / 2.0),
        "center_y": float((top + bottom) / 2.0),
    }


def _combine_points(*groups: list[tuple[float, float]]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for group in groups:
        points.extend(group)
    return points


def _landmark_point(landmarks, index: int) -> tuple[float, float]:
    point = landmarks[index]
    return float(point.x), float(point.y)


def _landmark_points(landmarks, indices: tuple[int, ...]) -> list[tuple[float, float]]:
    return [_landmark_point(landmarks, index) for index in indices]


def _midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _estimate_head_pose(landmarks) -> dict[str, float | str | bool]:
    left_eye_outer = _landmark_point(landmarks, 33)
    right_eye_outer = _landmark_point(landmarks, 263)
    nose_tip = _landmark_point(landmarks, 1)
    forehead = _landmark_point(landmarks, 10)
    chin = _landmark_point(landmarks, 152)
    mouth_left = _landmark_point(landmarks, 61)
    mouth_right = _landmark_point(landmarks, 291)

    face_width = max(math.dist(left_eye_outer, right_eye_outer), 1e-6)
    face_height = max(math.dist(forehead, chin), 1e-6)
    eye_center = _midpoint(left_eye_outer, right_eye_outer)
    mouth_center = _midpoint(mouth_left, mouth_right)
    face_center = _midpoint(eye_center, mouth_center)

    yaw = (nose_tip[0] - face_center[0]) / face_width
    pitch = (nose_tip[1] - face_center[1]) / face_height

    direction = "forward"
    if abs(yaw) >= HEAD_POSE_YAW_THRESH or abs(pitch) >= HEAD_POSE_PITCH_THRESH:
        if abs(yaw) >= abs(pitch):
            direction = "left" if yaw < 0 else "right"
        else:
            direction = "up" if pitch < 0 else "down"

    return {
        "yaw": float(yaw),
        "pitch": float(pitch),
        "direction": direction,
        "off_center": direction != "forward",
    }


def _head_pose_status(head_pose: dict[str, Any], drowsy: bool, yawning: bool) -> str:
    if bool(head_pose.get("alert")) or drowsy or yawning:
        return "risk"
    if bool(head_pose.get("off_center")):
        return "warning"
    return "normal"


def _mouth_status(mouth_ratio: Optional[float], mouth_alert: bool) -> str:
    if mouth_alert:
        return "risk"
    if mouth_ratio is not None and mouth_ratio >= YAWN_CLOSE_RATIO:
        return "warning"
    return "normal"


def _eyes_status(ear_val: Optional[float], drowsy: bool) -> str:
    if drowsy:
        return "risk"
    if ear_val is not None and ear_val <= (EYE_AR_THRESH + 0.02):
        return "warning"
    return "normal"


def _face_status(head_pose: dict[str, Any], drowsy: bool, yawning: bool) -> str:
    if bool(head_pose.get("alert")) or drowsy or yawning:
        return "risk"
    if bool(head_pose.get("off_center")):
        return "warning"
    return "normal"


def _stable_yawn_threshold() -> float:
    return max(_normalise_yawn_threshold(), YAWN_OPEN_RATIO)


def _update_head_pose_timer(
    *,
    head_pose: dict[str, Any],
    now: float,
    previous_started_at: Optional[float],
) -> tuple[Optional[float], float, bool]:
    if head_pose.get("off_center"):
        started_at = previous_started_at if previous_started_at is not None else now
        seconds = now - started_at
        alert = seconds >= HEAD_POSE_ALERT_SECONDS
        return started_at, seconds, alert

    return None, 0.0, False


def _update_yawn_state(
    *,
    mouth_ratio: Optional[float],
    now: float,
    previous_started_at: Optional[float],
    previous_frames: int,
) -> tuple[Optional[float], int, float, bool, bool]:
    if mouth_ratio is None:
        return None, 0, 0.0, False, False

    open_threshold = _stable_yawn_threshold()
    # Use a lower hold threshold once yawning starts so brief ratio jitter
    # does not reset the event before minimum duration/frames are reached.
    hold_threshold = min(open_threshold, YAWN_CLOSE_RATIO)

    should_count = mouth_ratio >= open_threshold
    if previous_started_at is not None and mouth_ratio >= hold_threshold:
        should_count = True

    if should_count:
        started_at = previous_started_at if previous_started_at is not None else now
        frames = previous_frames + 1
        seconds = now - started_at
        confirmed = frames >= YAWN_MIN_CONSEC_FRAMES
        alert = confirmed and seconds >= YAWN_MIN_DURATION_SECONDS
        return started_at, frames, seconds, confirmed, alert

    return None, 0, 0.0, False, False


def _prepare_mediapipe_env() -> None:
    cache_dir = Path(tempfile.gettempdir()) / "driver_safety_mpl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


def _ensure_face_landmarker_model() -> Optional[Path]:
    if _FACE_LANDMARKER_MODEL.is_file():
        return _FACE_LANDMARKER_MODEL

    try:
        _FACE_LANDMARKER_MODEL.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading face landmarker model to %s", _FACE_LANDMARKER_MODEL)
        with urlopen(_FACE_LANDMARKER_URL, timeout=20) as response:
            content = response.read()
        _FACE_LANDMARKER_MODEL.write_bytes(content)
        logger.info("Face landmarker model downloaded")
        return _FACE_LANDMARKER_MODEL
    except Exception as exc:
        logger.error("Failed to prepare face landmarker model: %s", exc)
        return None


def _select_backend() -> str:
    """Choose the safest available backend for the current runtime."""
    configured = os.getenv("DROWSINESS_BACKEND", "auto").strip().lower()
    if configured in {"opencv", "mediapipe"}:
        return configured

    if configured not in {"", "auto"}:
        logger.warning("Unknown DROWSINESS_BACKEND=%r. Falling back to auto mode.", configured)

    if _mediapipe_backend_available():
        return "mediapipe"

    return "opencv"


def _mediapipe_backend_available() -> bool:
    global _mediapipe_backend_supported

    if _mediapipe_backend_supported is not None:
        return _mediapipe_backend_supported

    _prepare_mediapipe_env()

    try:
        import mediapipe as mp

        model_path = _ensure_face_landmarker_model()
        if model_path is None:
            _mediapipe_backend_supported = False
            return False

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(model_path),
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        face_landmarker = FaceLandmarker.create_from_options(options)
        face_landmarker.close()
        _mediapipe_backend_supported = True
    except Exception as exc:
        logger.warning("MediaPipe backend probe failed, falling back to OpenCV: %s", exc)
        _mediapipe_backend_supported = False

    return bool(_mediapipe_backend_supported)


def _update_state(
    *,
    active: bool,
    backend: Optional[str],
    face_detected: bool,
    drowsy: bool,
    yawning: bool,
    ear: Optional[float],
    counter: int,
    head_pose: Optional[dict[str, Any]] = None,
    mouth: Optional[dict[str, Any]] = None,
    boxes: Optional[dict[str, Any]] = None,
    alert_message: Optional[str] = None,
) -> None:
    global _latest_frame_jpeg
    with _lock:
        payload = {
            "active": active,
            "backend": backend,
            "face_detected": face_detected,
            "drowsy": drowsy,
            "yawning": yawning,
            "ear": round(float(ear), 4) if isinstance(ear, (int, float)) else None,
            "counter": int(counter),
            "timestamp": time.time(),
        }
        if head_pose is not None:
            payload["head_pose"] = head_pose
        if mouth is not None:
            payload["mouth"] = mouth
        if boxes is not None:
            payload["boxes"] = boxes
        payload["alert_message"] = alert_message
        _state.update(payload)


def _set_inactive(backend: Optional[str]) -> None:
    _update_state(
        active=False,
        backend=backend,
        face_detected=False,
        drowsy=False,
        yawning=False,
        ear=None,
        counter=0,
        head_pose={
            "direction": "forward",
            "yaw": 0.0,
            "pitch": 0.0,
            "off_center": False,
            "alert": False,
            "seconds": 0.0,
        },
        mouth={
            "ratio": None,
            "smoothed_ratio": None,
            "open_seconds": 0.0,
        },
        boxes={"face": None, "eyes": None, "mouth": None},
        alert_message=None,
    )


def _store_frame(jpeg_bytes: Optional[bytes]) -> None:
    global _latest_frame_jpeg
    if jpeg_bytes is None:
        return
    with _lock:
        _latest_frame_jpeg = jpeg_bytes


# ── Core EAR calculation ────────────────────────────────────────────
def eye_aspect_ratio(eye):
    def _euclidean(p1, p2):
        return math.hypot((p1[0] - p2[0]), (p1[1] - p2[1]))

    A = _euclidean(eye[1], eye[5])
    B = _euclidean(eye[2], eye[4])
    C = _euclidean(eye[0], eye[3])
    if C <= 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)


def mouth_open_ratio(landmarks) -> float:
    top = landmarks[13]
    bottom = landmarks[14]
    left = landmarks[78]
    right = landmarks[308]

    vertical = math.hypot(top.x - bottom.x, top.y - bottom.y)
    horizontal = math.hypot(left.x - right.x, left.y - right.y)
    if horizontal <= 1e-6:
        return 0.0
    return vertical / horizontal


def _normalise_yawn_threshold() -> float:
    return YAWN_THRESH / 100.0 if YAWN_THRESH > 1.0 else YAWN_THRESH


def _update_eye_counter(counter: int, ear_val: Optional[float]) -> tuple[int, bool]:
    if ear_val is None:
        return 0, False

    if ear_val < EYE_AR_THRESH:
        counter += 1
    elif ear_val > (EYE_AR_THRESH + 0.02):
        counter = 0
    else:
        counter = max(0, counter - 1)

    return counter, counter >= EYE_AR_CONSEC_FRAMES


def _update_yawn_counter(counter: int, mouth_ratio: Optional[float]) -> tuple[int, bool]:
    if mouth_ratio is None:
        return 0, False

    if mouth_ratio >= _normalise_yawn_threshold():
        counter += 1
    else:
        counter = 0

    return counter, counter >= YAWN_CONSEC_FRAMES


def _estimate_mouth_open_metrics(mouth_roi_gray: np.ndarray) -> dict[str, float]:
    if mouth_roi_gray.size == 0:
        return {
            "score": 0.0,
            "area_ratio": 0.0,
            "width_ratio": 0.0,
            "height_ratio": 0.0,
            "openness": 0.0,
        }

    import cv2

    blurred = cv2.GaussianBlur(mouth_roi_gray, (5, 5), 0)
    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_h, roi_w = mask.shape[:2]
    roi_area = float(max(roi_h * roi_w, 1))
    best_metrics = {
        "score": 0.0,
        "area_ratio": 0.0,
        "width_ratio": 0.0,
        "height_ratio": 0.0,
        "openness": 0.0,
    }

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < roi_area * 0.01:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        center_x = x + (width / 2.0)
        center_y = y + (height / 2.0)
        if not (roi_w * 0.2 <= center_x <= roi_w * 0.8):
            continue
        if not (roi_h * 0.15 <= center_y <= roi_h * 0.85):
            continue

        area_ratio = area / roi_area
        width_ratio = width / max(roi_w, 1)
        height_ratio = height / max(roi_h, 1)
        openness = height / max(width, 1)
        score = area_ratio * (1.0 + (openness * 1.5)) * width_ratio * (1.0 + height_ratio)
        if score > best_metrics["score"]:
            best_metrics = {
                "score": float(score),
                "area_ratio": float(area_ratio),
                "width_ratio": float(width_ratio),
                "height_ratio": float(height_ratio),
                "openness": float(openness),
            }

    return best_metrics


def _estimate_mouth_open_score(mouth_roi_gray: np.ndarray) -> float:
    return _estimate_mouth_open_metrics(mouth_roi_gray)["score"]


def _opencv_yawn_ratio(mouth_roi_gray: np.ndarray) -> float:
    metrics = _estimate_mouth_open_metrics(mouth_roi_gray)
    area_ratio = metrics["area_ratio"]
    width_ratio = metrics["width_ratio"]
    height_ratio = metrics["height_ratio"]
    openness = metrics["openness"]

    # Yawns form a tall central opening. Wide but thin mouth contours
    # from neutral expressions or smiles should not trigger alerts.
    if area_ratio < 0.02:
        return 0.0
    if height_ratio < 0.14:
        return 0.0
    if openness < 0.25:
        return 0.0
    if not (0.18 <= width_ratio <= 0.80):
        return 0.0

    return min(
        1.0,
        (area_ratio * 5.0)
        + (height_ratio * 1.7)
        + (openness * 0.6),
    )


def _handle_alert_transitions(
    *,
    drowsy: bool,
    yawning: bool,
    ear_val: Optional[float],
    prev_drowsy: bool,
    prev_yawning: bool,
) -> None:
    global _last_event_log_ts

    now = time.time()
    if drowsy or yawning:
        if now - _last_event_log_ts >= 5:
            log_drowsiness_event(
                ear_score=float(ear_val or 0.0),
                yawning_detected=yawning,
            )
            _last_event_log_ts = now

    if drowsy:
        if not prev_drowsy:
            log_alert(user_id="system", alert_type="drowsiness", severity="high")
            start_alert_loop("drowsiness")
    else:
        stop_alert("drowsiness")

    if yawning:
        if not prev_yawning:
            log_alert(user_id="system", alert_type="yawning", severity="medium")
            start_alert_loop("yawning")
    else:
        stop_alert("yawning")


def _open_camera(cv2_module):
    candidate_backends = []
    for backend_name in ("CAP_AVFOUNDATION", "CAP_DSHOW", "CAP_MSMF", "CAP_ANY"):
        backend = getattr(cv2_module, backend_name, None)
        if isinstance(backend, int) and backend not in candidate_backends:
            candidate_backends.append(backend)

    for camera_index in (0, 1, 2, 3):
        for backend in candidate_backends:
            cap = cv2_module.VideoCapture(camera_index, backend)
            if cap.isOpened():
                logger.info("Opened webcam index %s using backend %s", camera_index, backend)
                return cap
            cap.release()

        cap = cv2_module.VideoCapture(camera_index)
        if cap.isOpened():
            logger.info("Opened webcam index %s using default backend", camera_index)
            return cap
        cap.release()

    logger.error("Cannot open any webcam index 0-3 — drowsiness detection disabled")
    return None


def _mediapipe_detection_loop() -> None:
    global _running

    backend = "mediapipe"
    _prepare_mediapipe_env()
    try:
        import cv2
        import mediapipe as mp
    except Exception as exc:
        logger.error("MediaPipe backend unavailable, drowsiness service disabled: %s", exc)
        _set_inactive(backend)
        return

    model_path = _ensure_face_landmarker_model()
    if model_path is None:
        _set_inactive(backend)
        return

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(model_path),
            delegate=BaseOptions.Delegate.CPU,
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    face_landmarker = FaceLandmarker.create_from_options(options)
    cap = _open_camera(cv2)

    if cap is None:
        face_landmarker.close()
        _set_inactive(backend)
        return

    counter = 0
    prev_drowsy = False
    prev_yawning = False
    prev_focus_alert = False
    frame_ts = 0
    ear_ema: Optional[float] = None
    mouth_ema: Optional[float] = None
    off_center_started_at: Optional[float] = None
    yawn_started_at: Optional[float] = None
    yawn_frames = 0
    smooth_face_box: Optional[dict[str, float]] = None
    smooth_eyes_box: Optional[dict[str, float]] = None
    smooth_mouth_box: Optional[dict[str, float]] = None
    pose_window: deque[dict[str, float]] = deque(maxlen=5)

    _update_state(
        active=True,
        backend=backend,
        face_detected=False,
        drowsy=False,
        yawning=False,
        ear=None,
        counter=0,
        head_pose={
            "direction": "forward",
            "yaw": 0.0,
            "pitch": 0.0,
            "off_center": False,
            "alert": False,
            "seconds": 0.0,
        },
        mouth={
            "ratio": None,
            "smoothed_ratio": None,
            "open_seconds": 0.0,
        },
        boxes={"face": None, "eyes": None, "mouth": None},
        alert_message=None,
    )
    logger.info("Webcam opened — drowsiness detection running with %s backend", backend)

    try:
        while _running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_ts += 33
            results = face_landmarker.detect_for_video(mp_image, frame_ts)

            face_detected = bool(results.face_landmarks)
            drowsy = False
            yawning = False
            ear_val = None
            mouth_ratio = None
            yawn_open_seconds = 0.0
            head_pose: dict[str, Any] = {
                "direction": "forward",
                "yaw": 0.0,
                "pitch": 0.0,
                "off_center": False,
                "alert": False,
                "seconds": 0.0,
            }
            boxes = {"face": None, "eyes": None, "mouth": None}
            alert_message = None
            now = time.time()

            if face_detected:
                landmarks = results.face_landmarks[0]

                left_eye_idx = [33, 160, 158, 133, 153, 144]
                right_eye_idx = [362, 385, 387, 263, 373, 380]

                left_eye = [
                    (landmarks[i].x, landmarks[i].y)
                    for i in left_eye_idx
                ]
                right_eye = [
                    (landmarks[i].x, landmarks[i].y)
                    for i in right_eye_idx
                ]

                raw_ear = (
                    eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)
                ) / 2.0
                ear_ema = _smooth_metric(ear_ema, raw_ear)
                ear_val = ear_ema
                counter, drowsy = _update_eye_counter(counter, ear_val)

                raw_mouth_ratio = mouth_open_ratio(landmarks)
                mouth_ema = _smooth_metric(mouth_ema, raw_mouth_ratio)
                mouth_ratio = mouth_ema

                yawn_started_at, yawn_frames, yawn_open_seconds, _, yawning = _update_yawn_state(
                    mouth_ratio=mouth_ratio,
                    now=now,
                    previous_started_at=yawn_started_at,
                    previous_frames=yawn_frames,
                )

                pose_now = _estimate_head_pose(landmarks)
                pose_window.append({
                    "yaw": float(pose_now["yaw"]),
                    "pitch": float(pose_now["pitch"]),
                })
                if pose_window:
                    avg_yaw = sum(sample["yaw"] for sample in pose_window) / len(pose_window)
                    avg_pitch = sum(sample["pitch"] for sample in pose_window) / len(pose_window)
                else:
                    avg_yaw = 0.0
                    avg_pitch = 0.0

                pose_direction = "forward"
                if abs(avg_yaw) >= HEAD_POSE_YAW_THRESH or abs(avg_pitch) >= HEAD_POSE_PITCH_THRESH:
                    if abs(avg_yaw) >= abs(avg_pitch):
                        pose_direction = "left" if avg_yaw < 0 else "right"
                    else:
                        pose_direction = "up" if avg_pitch < 0 else "down"

                pose_forward_yaw = HEAD_POSE_YAW_THRESH * HEAD_POSE_RETURN_RATIO
                pose_forward_pitch = HEAD_POSE_PITCH_THRESH * HEAD_POSE_RETURN_RATIO
                off_center = abs(avg_yaw) >= pose_forward_yaw or abs(avg_pitch) >= pose_forward_pitch
                if pose_direction == "forward" and abs(avg_yaw) < pose_forward_yaw and abs(avg_pitch) < pose_forward_pitch:
                    off_center = False

                head_pose = {
                    "direction": pose_direction,
                    "yaw": round(avg_yaw, 4),
                    "pitch": round(avg_pitch, 4),
                    "off_center": off_center,
                    "alert": False,
                    "seconds": 0.0,
                }
                off_center_started_at, off_center_seconds, pose_alert = _update_head_pose_timer(
                    head_pose=head_pose,
                    now=now,
                    previous_started_at=off_center_started_at,
                )
                head_pose["seconds"] = round(off_center_seconds, 2)
                head_pose["alert"] = pose_alert

                face_points = _landmark_points(landmarks, _FACE_LANDMARK_INDICES)
                eye_points = _combine_points(
                    _landmark_points(landmarks, _LEFT_EYE_INDICES),
                    _landmark_points(landmarks, _RIGHT_EYE_INDICES),
                )
                mouth_points = _landmark_points(landmarks, _MOUTH_INDICES)
                smooth_face_box = _smooth_box(smooth_face_box, _points_to_box(face_points, padding=0.03))
                smooth_eyes_box = _smooth_box(smooth_eyes_box, _points_to_box(eye_points, padding=0.02))
                smooth_mouth_box = _smooth_box(smooth_mouth_box, _points_to_box(mouth_points, padding=0.02))

                boxes = {
                    "face": _box_payload(smooth_face_box, _face_status(head_pose, drowsy, yawning)),
                    "eyes": _box_payload(smooth_eyes_box, _eyes_status(ear_val, drowsy)),
                    "mouth": _box_payload(smooth_mouth_box, _mouth_status(mouth_ratio, yawning)),
                }

                if head_pose["alert"]:
                    alert_message = "Focus on Road"
            else:
                counter = 0
                ear_ema = None
                mouth_ema = None
                off_center_started_at = None
                yawn_started_at = None
                yawn_frames = 0
                pose_window.clear()
                smooth_face_box = None
                smooth_eyes_box = None
                smooth_mouth_box = None

            success, jpeg = cv2.imencode(".jpg", frame)
            _store_frame(jpeg.tobytes() if success else None)
            focus_alert = bool(head_pose.get("alert"))
            _update_state(
                active=True,
                backend=backend,
                face_detected=face_detected,
                drowsy=drowsy,
                yawning=yawning,
                ear=ear_val,
                counter=counter,
                head_pose=head_pose,
                mouth={
                    "ratio": round(float(raw_mouth_ratio), 4) if face_detected else None,
                    "smoothed_ratio": round(float(mouth_ratio), 4) if mouth_ratio is not None else None,
                    "open_seconds": round(float(yawn_open_seconds), 2),
                },
                boxes=boxes,
                alert_message=alert_message,
            )

            _handle_alert_transitions(
                drowsy=drowsy,
                yawning=yawning,
                ear_val=ear_val,
                prev_drowsy=prev_drowsy,
                prev_yawning=prev_yawning,
            )
            if focus_alert:
                if not prev_focus_alert:
                    log_alert(user_id="system", alert_type="focus_road", severity="high")
                start_alert_loop("focus")
            else:
                stop_alert("focus")

            prev_drowsy = drowsy
            prev_yawning = yawning
            prev_focus_alert = focus_alert

            time.sleep(0.03)
    except Exception as exc:
        logger.error("Detection loop error: %s", exc)
    finally:
        cap.release()
        face_landmarker.close()
        stop_alert("drowsiness")
        stop_alert("yawning")
        stop_alert("focus")
        _set_inactive(backend)
        logger.info("Webcam released — detection stopped")


def _opencv_detection_loop() -> None:
    global _running

    backend = "opencv"
    try:
        import cv2
    except Exception as exc:
        logger.error("OpenCV backend unavailable, drowsiness service disabled: %s", exc)
        _set_inactive(backend)
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
    )
    if face_cascade.empty() or eye_cascade.empty():
        logger.error("OpenCV cascades could not be loaded")
        _set_inactive(backend)
        return

    cap = _open_camera(cv2)
    if cap is None:
        _set_inactive(backend)
        return

    counter = 0
    prev_drowsy = False
    prev_yawning = False
    eye_ema: Optional[float] = None
    mouth_ema: Optional[float] = None
    yawn_started_at: Optional[float] = None
    yawn_frames = 0
    smooth_face_box: Optional[dict[str, float]] = None
    smooth_eyes_box: Optional[dict[str, float]] = None
    smooth_mouth_box: Optional[dict[str, float]] = None

    _update_state(
        active=True,
        backend=backend,
        face_detected=False,
        drowsy=False,
        yawning=False,
        ear=None,
        counter=0,
        head_pose={
            "direction": "forward",
            "yaw": 0.0,
            "pitch": 0.0,
            "off_center": False,
            "alert": False,
            "seconds": 0.0,
        },
        mouth={
            "ratio": None,
            "smoothed_ratio": None,
            "open_seconds": 0.0,
        },
        boxes={"face": None, "eyes": None, "mouth": None},
        alert_message=None,
    )
    logger.info("Webcam opened — drowsiness detection running with %s backend", backend)

    try:
        while _running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(120, 120),
            )

            face_detected = len(faces) > 0
            drowsy = False
            yawning = False
            ear_val = None
            mouth_ratio = None
            raw_mouth_ratio = None
            yawn_open_seconds = 0.0
            head_pose = {
                "direction": "forward",
                "yaw": 0.0,
                "pitch": 0.0,
                "off_center": False,
                "alert": False,
                "seconds": 0.0,
            }
            boxes = {"face": None, "eyes": None, "mouth": None}
            now = time.time()

            if face_detected:
                x, y, w, h = max(faces, key=lambda bbox: bbox[2] * bbox[3])
                roi_gray = gray[y : y + h, x : x + w]
                eye_band = roi_gray[: max(int(h * 0.5), 1), :]
                eyes = eye_cascade.detectMultiScale(
                    eye_band,
                    scaleFactor=1.1,
                    minNeighbors=6,
                    minSize=(25, 25),
                )

                left_hits = 0
                right_hits = 0
                mid_x = eye_band.shape[1] / 2.0
                for ex, _, ew, _ in eyes:
                    eye_center_x = ex + (ew / 2.0)
                    if eye_center_x < mid_x:
                        left_hits += 1
                    else:
                        right_hits += 1

                visible_eyes = int(left_hits > 0) + int(right_hits > 0)
                if visible_eyes >= 2:
                    raw_ear = 0.34
                elif visible_eyes == 1:
                    raw_ear = 0.22
                else:
                    raw_ear = 0.16

                eye_ema = _smooth_metric(eye_ema, raw_ear, alpha=0.5)
                ear_val = eye_ema
                counter, drowsy = _update_eye_counter(counter, ear_val)

                mouth_top = int(h * 0.55)
                mouth_left = int(w * 0.22)
                mouth_bottom = int(h * 0.92)
                mouth_right = int(w * 0.78)
                mouth_roi = roi_gray[mouth_top:mouth_bottom, mouth_left:mouth_right]
                raw_mouth_ratio = _opencv_yawn_ratio(mouth_roi)
                mouth_ema = _smooth_metric(mouth_ema, raw_mouth_ratio)
                mouth_ratio = mouth_ema
                yawn_started_at, yawn_frames, yawn_open_seconds, _, yawning = _update_yawn_state(
                    mouth_ratio=mouth_ratio,
                    now=now,
                    previous_started_at=yawn_started_at,
                    previous_frames=yawn_frames,
                )

                face_points = [
                    (x / 640.0, y / 480.0),
                    ((x + w) / 640.0, y / 480.0),
                    ((x + w) / 640.0, (y + h) / 480.0),
                    (x / 640.0, (y + h) / 480.0),
                ]
                smooth_face_box = _smooth_box(smooth_face_box, _points_to_box(face_points, padding=0.0))

                eye_points: list[tuple[float, float]] = []
                for ex, ey, ew, eh in eyes:
                    ex0 = (x + ex) / 640.0
                    ey0 = (y + ey) / 480.0
                    ex1 = (x + ex + ew) / 640.0
                    ey1 = (y + ey + eh) / 480.0
                    eye_points.extend([(ex0, ey0), (ex1, ey0), (ex1, ey1), (ex0, ey1)])
                smooth_eyes_box = _smooth_box(smooth_eyes_box, _points_to_box(eye_points, padding=0.0))

                mouth_points = [
                    ((x + mouth_left) / 640.0, (y + mouth_top) / 480.0),
                    ((x + mouth_right) / 640.0, (y + mouth_top) / 480.0),
                    ((x + mouth_right) / 640.0, (y + mouth_bottom) / 480.0),
                    ((x + mouth_left) / 640.0, (y + mouth_bottom) / 480.0),
                ]
                smooth_mouth_box = _smooth_box(smooth_mouth_box, _points_to_box(mouth_points, padding=0.0))

                boxes = {
                    "face": _box_payload(smooth_face_box, _face_status(head_pose, drowsy, yawning)),
                    "eyes": _box_payload(smooth_eyes_box, _eyes_status(ear_val, drowsy)),
                    "mouth": _box_payload(smooth_mouth_box, _mouth_status(mouth_ratio, yawning)),
                }
            else:
                counter = 0
                eye_ema = None
                mouth_ema = None
                yawn_started_at = None
                yawn_frames = 0
                smooth_face_box = None
                smooth_eyes_box = None
                smooth_mouth_box = None

            success, jpeg = cv2.imencode(".jpg", frame)
            _store_frame(jpeg.tobytes() if success else None)
            _update_state(
                active=True,
                backend=backend,
                face_detected=face_detected,
                drowsy=drowsy,
                yawning=yawning,
                ear=ear_val,
                counter=counter,
                head_pose=head_pose,
                mouth={
                    "ratio": round(float(raw_mouth_ratio), 4) if raw_mouth_ratio is not None else None,
                    "smoothed_ratio": round(float(mouth_ratio), 4) if mouth_ratio is not None else None,
                    "open_seconds": round(float(yawn_open_seconds), 2),
                },
                boxes=boxes,
                alert_message=None,
            )

            _handle_alert_transitions(
                drowsy=drowsy,
                yawning=yawning,
                ear_val=ear_val,
                prev_drowsy=prev_drowsy,
                prev_yawning=prev_yawning,
            )
            prev_drowsy = drowsy
            prev_yawning = yawning

            time.sleep(0.03)
    except Exception as exc:
        logger.error("OpenCV fallback loop error: %s", exc)
    finally:
        cap.release()
        stop_alert("drowsiness")
        stop_alert("yawning")
        _set_inactive(backend)
        logger.info("Webcam released — detection stopped")


# ── Public API ───────────────────────────────────────────────────────
def start():
    """Start the background detection thread."""
    global _running, _thread

    if _thread is not None and _thread.is_alive():
        logger.info("Drowsiness detection thread already running")
        return

    backend = _select_backend()
    target = _mediapipe_detection_loop if backend == "mediapipe" else _opencv_detection_loop

    if backend == "opencv" and tuple(sys.version_info[:2]) >= (3, 13):
        logger.warning(
            "Using OpenCV fallback for drowsiness detection on Python %s.%s",
            sys.version_info[0],
            sys.version_info[1],
        )

    _running = True
    _thread = threading.Thread(
        target=target,
        name=f"drowsiness-{backend}",
        daemon=True,
    )
    _thread.start()
    logger.info("Drowsiness detection thread started using %s backend", backend)


def stop():
    """Signal the detection loop to stop."""
    global _running, _thread

    _running = False
    stop_alert("drowsiness")
    stop_alert("yawning")
    stop_alert("focus")
    logger.info("Drowsiness detection stopping…")

    thread = _thread
    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=2.0)
    _thread = None


def get_state() -> dict:
    """Return a snapshot of the current detection state."""
    with _lock:
        return _state.copy()


def get_frame() -> Optional[bytes]:
    """Return the latest webcam frame as JPEG bytes."""
    with _lock:
        return _latest_frame_jpeg
