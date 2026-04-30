"""
Fog / Visibility Detection Service.

Loads the EfficientNet-B0 fog detection model ONCE at startup
and provides a predict() function for inference.

Model logic is identical to the original fog_detection/app.py:
  - Same model architecture (timm efficientnet_b0, 2 classes)
  - Same transforms (Resize 224, ToTensor, Normalize)
  - Same prediction logic

The model weights file (fog_model.pth) is stored in backend/models/.
"""

from typing import Optional
import io
import time

try:
    from PIL import Image, ImageStat, ImageFilter
    import torch
    import timm
    from torchvision import transforms
except Exception:
    Image = None
    ImageStat = None
    ImageFilter = None
    torch = None
    timm = None
    transforms = None

from config import FOG_MODEL_PATH, FOG_MODEL_CLASSES
from database.mongo import log_alert, log_fog_prediction
from utils.logger import get_logger

logger = get_logger("fog_service")

# ── Model (loaded once) ─────────────────────────────────────────────
_device = torch.device("cpu") if torch is not None else None
_model = None
_transform = (
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    if transforms is not None
    else None
)

# ── Track last prediction for state queries ──────────────────────────
_last_state: dict = {"active": False}
_last_fog_alert_ts = 0.0


def load_model():
    """Load the fog detection model into memory. Call once at startup."""
    global _model
    if torch is None or timm is None or _transform is None:
        logger.error("Fog service dependencies are missing")
        _model = None
        return

    try:
        _model = timm.create_model(
            "efficientnet_b0", pretrained=False, num_classes=FOG_MODEL_CLASSES
        )
        _model.load_state_dict(torch.load(FOG_MODEL_PATH, map_location=_device))
        _model.eval()
        logger.info(f"Fog detection model loaded from {FOG_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load fog model: {e}")
        _model = None


def _severity_from_probability(fog_probability: float) -> str:
    if fog_probability >= 0.75:
        return "high"
    if fog_probability >= 0.5:
        return "medium"
    return "low"


def _fallback_visibility_from_image(image_bytes: bytes) -> dict:
    """Estimate fog/clear conditions from basic image statistics when the model is unavailable."""
    if Image is None or ImageStat is None:
        return {
            "active": False,
            "error": "PIL unavailable",
        }

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        resized = image.resize((224, 224))
        stat = ImageStat.Stat(resized)
        brightness = sum(stat.mean) / (3.0 * 255.0)
        contrast = sum(stat.stddev) / (3.0 * 255.0)

        # saturation from HSV — fog tends to desaturate backgrounds
        hsv = resized.convert("HSV")
        hsv_stat = ImageStat.Stat(hsv)
        saturation_mean = hsv_stat.mean[1] / 255.0

        # edge density proxy: fewer edges when foggy
        edges = resized.convert("L").filter(ImageFilter.FIND_EDGES)
        edges_stat = ImageStat.Stat(edges)
        edge_mean = edges_stat.mean[0] / 255.0

        # blur proxy (variance on grayscale)
        grayscale = resized.convert("L")
        blur_stat = ImageStat.Stat(grayscale)
        blur_proxy = blur_stat.var[0] / (255.0 * 255.0)

        # Heuristic scoring — tuned to be more sensitive to desaturation and low edge density
        fog_score = 0.0
        if saturation_mean < 0.28:
            fog_score += 0.50
        if contrast < 0.14:
            fog_score += 0.20
        if edge_mean < 0.06:
            fog_score += 0.20
        if blur_proxy < 0.01:
            fog_score += 0.10

        # Log diagnostic values to help tuning
        logger.debug(
            "Fog fallback stats - brightness=%.3f contrast=%.3f sat=%.3f edges=%.3f blur=%.6f",
            brightness,
            contrast,
            saturation_mean,
            edge_mean,
            blur_proxy,
        )

        fog_score = min(0.98, fog_score)
        label = "Fog/Smog" if fog_score >= 0.5 else "Clear"
        confidence = 0.60 + abs(fog_score - 0.5) * 0.8
        confidence = min(0.99, max(0.55, confidence))

        return {
            "active": True,
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "fog_probability": round(fog_score, 4),
            "fallback": True,
        }
    except Exception as exc:
        logger.error(f"Fog fallback prediction error: {exc}")
        return {"active": False, "error": str(exc)}


def predict(image_bytes: bytes, user_id: str = "system", image_name: str = "camera_frame.jpg") -> dict:
    """
    Run fog/visibility prediction on raw image bytes.
    Returns dict with prediction, confidence, and active status.
    """
    global _last_state

    if _model is None:
        _last_state = _fallback_visibility_from_image(image_bytes)
        return _last_state

    try:
        if Image is None or torch is None or _transform is None:
            _last_state = {"active": False, "error": "ML dependencies unavailable"}
            return _last_state

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = _transform(image).unsqueeze(0)

        with torch.no_grad():
            output = _model(img_tensor)
            prob = torch.softmax(output, dim=1)
            confidence = prob.max().item()
            _, pred = torch.max(output, 1)

        label = "Clear" if pred.item() == 0 else "Fog/Smog"
        fog_probability = float(prob[0][1].item()) if prob.shape[1] > 1 else 0.0

        log_fog_prediction(image_name=image_name, fog_probability=fog_probability)

        global _last_fog_alert_ts
        if label == "Fog/Smog":
            now = time.time()
            if now - _last_fog_alert_ts >= 20:
                log_alert(
                    user_id=user_id,
                    alert_type="fog",
                    severity=_severity_from_probability(fog_probability),
                )
                _last_fog_alert_ts = now

        _last_state = {
            "active": True,
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "fog_probability": round(fog_probability, 4),
        }
        return _last_state

    except Exception as e:
        logger.error(f"Fog prediction error: {e}")
        _last_state = {"active": False, "error": str(e)}
        return _last_state


def get_state() -> dict:
    """Return the last prediction state."""
    return _last_state.copy()
