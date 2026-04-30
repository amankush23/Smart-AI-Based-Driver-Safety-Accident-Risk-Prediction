"""
API Routes — REST endpoints for the unified system.
"""

import io
import time

_start_time = time.time()

from fastapi import APIRouter, UploadFile, File, Response, Depends
from pydantic import BaseModel

from database.mongo import get_alerts, get_drowsiness_events
from services.auth_service import get_current_user
from services import drowsiness_service, fog_service, stress_service, visibility_service, kid_safety_service
from services import accident_service
from services.analytics_service import generate_summary
from services.risk_engine import compute_unified_risk
from utils.logger import get_logger

logger = get_logger("routes.api")
router = APIRouter(prefix="/api")


@router.get("/status")
def get_status():
    """System health — reports status of all modules."""
    try:
        d_state = drowsiness_service.get_state()
        f_state = fog_service.get_state()
        s_state = stress_service.get_state()
        v_state = visibility_service.get_state()
        k_state = kid_safety_service.get_state()
        risk = compute_unified_risk(d_state, f_state, s_state, v_state, k_state)
        return {
            "service": "driver-safety-system",
            "status": "online",
            "version": "2.0.0",
            "timestamp": time.time(),
            "uptime": time.time() - _start_time,
            "modules": {
                "drowsiness": {"active": d_state.get("active", False)},
                "fog": {"active": f_state.get("active", False)},
                "stress": {"active": s_state.get("active", False)},
                "visibility": {"active": v_state.get("active", False)},
                "motion_detection": {
                    "active": v_state.get("active", False),
                    "engine_on": v_state.get("child_presence", {}).get("engine_on", True),
                },
                "kid_safety": {
                    "active": k_state.get("active", False),
                    "status": k_state.get("status", "NO_FACE"),
                },
            },
            "risk_score": risk.get("overall_score", 0),
            "risk_level": risk.get("risk_level", "low"),
        }
    except Exception as e:
        logger.error(f"Status error: {e}")
        return {"error": str(e)}


@router.get("/risk")
def get_risk():
    """Unified risk assessment from all modules."""
    try:
        d_state = drowsiness_service.get_state()
        f_state = fog_service.get_state()
        s_state = stress_service.get_state()
        v_state = visibility_service.get_state()
        k_state = kid_safety_service.get_state()
        result = compute_unified_risk(d_state, f_state, s_state, v_state, k_state)
        result["timestamp"] = time.time()
        return result
    except Exception as e:
        logger.error(f"Risk endpoint error: {e}")
        return {"error": str(e)}


@router.get("/drowsiness")
def get_drowsiness():
    """Current drowsiness/yawn detection state."""
    try:
        return drowsiness_service.get_state()
    except Exception as e:
        logger.error(f"Drowsiness endpoint error: {e}")
        return {"error": str(e)}


@router.get("/drowsiness/logs")
def get_drowsiness_logs(user: dict = Depends(get_current_user)):
    """Protected endpoint to fetch drowsiness events log."""
    try:
        return {"events": get_drowsiness_events(limit=200)}
    except Exception as e:
        logger.error(f"Drowsiness logs error: {e}")
        return {"error": str(e)}


@router.get("/fog")
def get_fog(user: dict = Depends(get_current_user)):
    """Current fog detection state."""
    try:
        return fog_service.get_state()
    except Exception as e:
        logger.error(f"Fog endpoint error: {e}")
        return {"error": str(e)}


@router.get("/stress")
def get_stress_state(user: dict = Depends(get_current_user)):
    """Current stress detection state."""
    try:
        return stress_service.get_state()
    except Exception as e:
        logger.error(f"Stress endpoint error: {e}")
        return {"error": str(e)}


@router.post("/stress/upload")
async def upload_stress_audio(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """Upload an audio sample and run stress detection."""
    try:
        if not file.content_type or not file.content_type.startswith("audio/"):
            return {"error": "Invalid audio format"}
        contents = await file.read()
        return stress_service.predict_from_bytes(contents, filename=file.filename or "sample.wav")
    except Exception as e:
        logger.error(f"Stress upload error: {e}")
        return {"error": str(e)}


@router.get("/visibility")
def get_visibility_state(user: dict = Depends(get_current_user)):
    """Current camera visibility and child-presence state."""
    try:
        return visibility_service.get_state()
    except Exception as e:
        logger.error(f"Visibility endpoint error: {e}")
        return {"error": str(e)}


@router.post("/visibility/predict-frame")
async def predict_visibility_from_camera(user: dict = Depends(get_current_user)):
    """Analyze latest webcam frame for visibility and child-presence cues."""
    try:
        frame = drowsiness_service.get_frame()
        if frame is None:
            return {"error": "No camera frame available"}
        return visibility_service.predict(frame, user_id=user["id"], image_name="camera_frame.jpg")
    except Exception as e:
        logger.error(f"Visibility predict-frame error: {e}")
        return {"error": str(e)}


@router.get("/motion-detection")
def get_motion_detection_state(user: dict = Depends(get_current_user)):
    """Current motion detection state."""
    try:
        return visibility_service.get_state().get("child_presence", {})
    except Exception as e:
        logger.error(f"Motion detection endpoint error: {e}")
        return {"error": str(e)}


def read_kid_safety_state(user_id: str = "system") -> dict:
    """Read or refresh kid-safety state from the latest camera frame."""
    frame = drowsiness_service.get_frame()
    if frame is None:
        return kid_safety_service.get_state()
    return kid_safety_service.predict(frame, user_id=user_id, image_name="camera_frame.jpg")


@router.get("/kid-safety")
def get_kid_safety_state(user: dict = Depends(get_current_user)):
    """Current kid safety detection state."""
    try:
        return read_kid_safety_state(user_id=user["id"])
    except Exception as e:
        logger.error(f"Kid safety endpoint error: {e}")
        return {"error": str(e)}


@router.post("/motion-detection/engine")
def set_motion_engine(on: bool, user: dict = Depends(get_current_user)):
    """Toggle engine state for motion detection logic."""
    try:
        return visibility_service.set_engine(on)
    except Exception as e:
        logger.error(f"Motion detection engine toggle error: {e}")
        return {"error": str(e)}


@router.get("/frame")
def get_frame():
    """Latest webcam frame as JPEG (for testing / fog forwarding)."""
    try:
        frame = drowsiness_service.get_frame()
        if frame is None:
            return Response(status_code=503, content='{"error":"No frame available"}', media_type="application/json")
        return Response(content=frame, media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Frame endpoint error: {e}")
        return {"error": str(e)}


@router.post("/fog/upload")
async def upload_fog_image(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """Upload an image for fog detection (manual / testing)."""
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            return {"error": "Invalid image format"}
        contents = await file.read()
        result = fog_service.predict(contents, user_id=user["id"], image_name=file.filename or "upload.jpg")
        return result
    except Exception as e:
        logger.error(f"Fog upload error: {e}")
        return {"error": str(e)}


@router.post("/fog/predict-frame")
async def predict_from_camera(user: dict = Depends(get_current_user)):
    """Grab the latest camera frame and run fog detection on it."""
    try:
        frame = drowsiness_service.get_frame()
        if frame is None:
            return {"error": "No camera frame available"}
        return fog_service.predict(frame, user_id=user["id"], image_name="camera_frame.jpg")
    except Exception as e:
        logger.error(f"Fog predict-frame error: {e}")
        return {"error": str(e)}


@router.get("/alerts")
def get_alert_history(user: dict = Depends(get_current_user)):
    """Protected endpoint for alert history table data."""
    try:
        alerts = get_alerts(user_id=user["id"], limit=200)
        return {"alerts": alerts}
    except Exception as e:
        logger.error(f"Alert history error: {e}")
        return {"error": str(e)}


@router.get("/analytics/summary")
def analytics_summary(user: dict = Depends(get_current_user)):
    """Protected endpoint for analytics summary and safety score."""
    try:
        return generate_summary(user_id=user["id"])
    except Exception as e:
        logger.error(f"Analytics summary error: {e}")
        return {"error": str(e)}


# ── Accident Severity Prediction ─────────────────────────────────────

class AccidentInput(BaseModel):
    State: str
    City: str
    No_of_Vehicles: int
    Road_Type: str
    Road_Surface: str
    Light_Condition: str
    Weather: str
    Casualty_Class: str
    Casualty_Sex: str
    Casualty_Age: int
    Vehicle_Type: str


@router.post("/accident/predict")
def predict_accident(data: AccidentInput):
    """Predict road accident severity using the XGBoost model."""
    try:
        result = accident_service.predict(data.model_dump())
        return result
    except Exception as e:
        logger.error(f"Accident prediction error: {e}")
        return {"error": str(e)}


@router.get("/accident/status")
def accident_status():
    """Check if accident prediction model is loaded."""
    try:
        return {"loaded": accident_service.is_loaded()}
    except Exception as e:
        logger.error(f"Accident status error: {e}")
        return {"error": str(e)}
