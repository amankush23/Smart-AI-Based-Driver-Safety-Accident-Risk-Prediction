"""
Unified Driver Risk Score Engine.

Combines outputs from drowsiness detection and fog detection
into a single risk assessment.

Risk Levels:
    0–30   → Low      (Green)
   31–60   → Moderate (Yellow)
   61–80   → High     (Orange)
   81–100  → Critical (Red)

Weighting:
    60% drowsiness (immediate driver safety)
    40% fog        (environmental hazard)
"""

from config import (
    CHILD_WEIGHT,
    DROWSINESS_WEIGHT,
    FOG_WEIGHT,
    KID_SAFETY_WEIGHT,
    STRESS_WEIGHT,
    VISIBILITY_WEIGHT,
)
from typing import Optional
from utils.logger import get_logger

logger = get_logger("risk_engine")


def calculate_drowsiness_risk(state: dict) -> float:
    """Risk score (0–100) from drowsiness detection state."""
    if not state or not state.get("active"):
        return 0.0
    if not state.get("face_detected"):
        return 0.0
    head_pose = state.get("head_pose") or {}
    if head_pose.get("alert"):
        return 78.0
    if state.get("drowsy"):
        return 90.0
    if state.get("yawning"):
        return 55.0
    ear = state.get("ear")
    if isinstance(ear, (int, float)) and ear > 0 and ear < 0.30:
        return min(45.0, 25.0 + (0.30 - ear) * 400)
    return 10.0


def calculate_fog_risk(state: dict) -> float:
    """Risk score (0–100) from fog detection state."""
    if not state or not state.get("active"):
        return 0.0
    if state.get("prediction") == "Fog/Smog":
        return min(95.0, state.get("confidence", 50.0))
    else:
        return max(5.0, 100.0 - state.get("confidence", 50.0))


def calculate_stress_risk(state: dict) -> float:
    """Risk score (0-100) from stress detection state."""
    if not state or not state.get("active"):
        return 0.0
    return min(100.0, max(0.0, float(state.get("score", 0.0))))


def calculate_visibility_risk(state: dict) -> float:
    """Risk score (0-100) from camera visibility conditions."""
    if not state or not state.get("active"):
        return 0.0
    vis = state.get("visibility", {})
    return min(100.0, max(0.0, float(vis.get("score", 0.0))))


def calculate_child_risk(state: dict) -> float:
    """Risk score (0-100) from child-presence detection."""
    if not state or not state.get("active"):
        return 0.0
    child = state.get("child_presence", {})
    if child.get("alert"):
        return 100.0
    return min(100.0, max(0.0, float(child.get("score", 0.0))))


def calculate_kid_safety_risk(state: dict) -> float:
    """Risk score (0-100) from kid safety detection."""
    if not state or not state.get("active"):
        return 0.0
    status = str(state.get("status", "NO_FACE")).upper()
    if status == "SAFE":
        return min(10.0, max(0.0, float(state.get("risk", 5.0))))
    if status == "WARNING":
        return max(35.0, min(60.0, float(state.get("risk", 40.0))))
    if status == "DANGER":
        return max(85.0, min(100.0, float(state.get("risk", 95.0))))
    if status == "NO_FACE":
        return 0.0
    return min(100.0, max(0.0, float(state.get("risk", 0.0))))


def get_risk_level(score: float) -> str:
    """Classify numeric risk into named level."""
    if score >= 80:
        return "critical"
    if score >= 60:
        return "high"
    if score >= 35:
        return "moderate"
    return "low"


def compute_unified_risk(
    drowsiness_state: dict,
    fog_state: dict,
    stress_state: Optional[dict] = None,
    visibility_state: Optional[dict] = None,
    kid_safety_state: Optional[dict] = None,
) -> dict:
    """
    Compute the unified Driver Risk Score.
    
    Returns comprehensive risk assessment dict.
    """
    d_risk = calculate_drowsiness_risk(drowsiness_state)
    f_risk = calculate_fog_risk(fog_state)
    s_risk = calculate_stress_risk(stress_state or {})
    v_risk = calculate_visibility_risk(visibility_state or {})
    c_risk = calculate_child_risk(visibility_state or {})
    k_risk = calculate_kid_safety_risk(kid_safety_state or {})

    d_active = bool(drowsiness_state and drowsiness_state.get("active"))
    f_active = bool(fog_state and fog_state.get("active"))
    s_active = bool(stress_state and stress_state.get("active"))
    v_active = bool(visibility_state and visibility_state.get("active"))
    c_active = bool(visibility_state and visibility_state.get("active"))
    k_active = bool(kid_safety_state and kid_safety_state.get("active"))

    raw_weights = {
        "drowsiness": DROWSINESS_WEIGHT,
        "fog": FOG_WEIGHT,
        "stress": STRESS_WEIGHT,
        "visibility": VISIBILITY_WEIGHT,
        "child": CHILD_WEIGHT,
        "kid_safety": KID_SAFETY_WEIGHT,
    }
    weight_sum = sum(raw_weights.values())
    if weight_sum <= 0:
        raw_weights = {
            "drowsiness": 0.35,
            "fog": 0.25,
            "stress": 0.20,
            "visibility": 0.10,
            "child": 0.10,
        }
        weight_sum = 1.0
    weights = {k: (v / weight_sum) for k, v in raw_weights.items()}

    active_risks = []
    if d_active:
        active_risks.append(d_risk * weights["drowsiness"])
    if f_active:
        active_risks.append(f_risk * weights["fog"])
    if s_active:
        active_risks.append(s_risk * weights["stress"])
    if v_active:
        active_risks.append(v_risk * weights["visibility"])
    if c_active:
        active_risks.append(c_risk * weights["child"])
    if k_active:
        active_risks.append(k_risk * weights["kid_safety"])

    unified = sum(active_risks) if active_risks else 0.0
    if k_active and str((kid_safety_state or {}).get("status", "")).upper() == "DANGER":
        unified += 15.0

    unified = min(100.0, unified)

    return {
        "overall_score": round(unified, 1),
        "risk_level": get_risk_level(unified),
        "drowsiness": {
            "active": d_active,
            "risk_score": round(d_risk, 1),
            "face_detected": drowsiness_state.get("face_detected", False) if d_active else False,
            "drowsy": drowsiness_state.get("drowsy", False) if d_active else False,
            "yawning": drowsiness_state.get("yawning", False) if d_active else False,
            "ear": drowsiness_state.get("ear", 0) if d_active else None,
            "head_pose": drowsiness_state.get("head_pose", {}) if d_active else {},
            "mouth": drowsiness_state.get("mouth", {}) if d_active else {},
            "boxes": drowsiness_state.get("boxes", {}) if d_active else {},
            "alert_message": drowsiness_state.get("alert_message") if d_active else None,
        },
        "fog": {
            "active": f_active,
            "risk_score": round(f_risk, 1),
            "prediction": fog_state.get("prediction", "N/A") if f_active else "N/A",
            "confidence": fog_state.get("confidence", 0) if f_active else None,
        },
        "stress": {
            "active": s_active,
            "risk_score": round(s_risk, 1),
            "level": (stress_state or {}).get("level", "Normal"),
            "confidence": (stress_state or {}).get("confidence", 0.0),
            "source": (stress_state or {}).get("source", "idle"),
        },
        "visibility": {
            "active": v_active,
            "risk_score": round(v_risk, 1),
            "condition": (visibility_state or {}).get("visibility", {}).get("condition", "Unknown"),
            "brightness": (visibility_state or {}).get("visibility", {}).get("brightness", 0.0),
            "contrast": (visibility_state or {}).get("visibility", {}).get("contrast", 0.0),
            "blur_var": (visibility_state or {}).get("visibility", {}).get("blur_var", 0.0),
        },
        "motion_detection": {
            "active": c_active,
            "risk_score": round(c_risk, 1),
            "engine_on": (visibility_state or {}).get("child_presence", {}).get("engine_on", True),
            "motion": (visibility_state or {}).get("child_presence", {}).get("motion", False),
            "alert": (visibility_state or {}).get("child_presence", {}).get("alert", False),
            "recent_pct": (visibility_state or {}).get("child_presence", {}).get("recent_pct", 0.0),
        },
        "kid_safety": {
            "active": k_active,
            "risk_score": round(k_risk, 1),
            "kid_detected": (kid_safety_state or {}).get("kid_detected", False) if k_active else False,
            "adult_present": (kid_safety_state or {}).get("adult_present", False) if k_active else False,
            "status": (kid_safety_state or {}).get("status", "NO_FACE") if k_active else "NO_FACE",
            "message": (kid_safety_state or {}).get("message", "No occupant detected") if k_active else "No occupant detected",
            "alone_seconds": (kid_safety_state or {}).get("alone_seconds", 0.0) if k_active else 0.0,
        },
        "active_modules": int(d_active) + int(f_active) + int(s_active) + int(v_active) + int(c_active) + int(k_active),
    }
