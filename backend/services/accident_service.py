"""
Road Accident Severity Prediction Service.

Loads a pre-trained XGBoost model + LabelEncoder (joblib)
and exposes a predict() function used by the API route.
"""

import joblib
import pandas as pd
import json
import subprocess
import sys
from pathlib import Path

from config import MODELS_DIR
from utils.logger import get_logger

logger = get_logger("accident_service")

# ── Model paths ──────────────────────────────────────────────────────
_MODEL_PATH = MODELS_DIR / "accident_prediction_model.pkl"
_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

# ── Module-level references (loaded once) ────────────────────────────
_model = None
_label_encoder = None


def _run_subprocess_prediction(input_data: dict | None) -> dict:
    """Execute model loading/prediction in a subprocess to isolate native crashes."""
    worker_code = """
import json
import sys
import joblib
import pandas as pd

model_path = sys.argv[1]
encoder_path = sys.argv[2]
payload = json.loads(sys.stdin.read())

model = joblib.load(model_path)
encoder = joblib.load(encoder_path)

if payload is None:
    print(json.dumps({"loaded": True}))
else:
    frame = pd.DataFrame([payload])
    raw = model.predict(frame)
    severity = encoder.inverse_transform(raw)[0]
    print(json.dumps({"prediction": str(severity)}))
"""

    process = subprocess.run(
        [sys.executable, "-c", worker_code, str(_MODEL_PATH), str(_ENCODER_PATH)],
        input=json.dumps(input_data),
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or "Subprocess accident prediction failed")

    output = process.stdout.strip()
    if not output:
        raise RuntimeError("Subprocess accident prediction produced no output")
    return json.loads(output)


def load_model():
    """Load the accident prediction model and label encoder from disk."""
    global _model, _label_encoder
    if sys.version_info >= (3, 14):
        # Keep in-process model disabled on 3.14; handled via subprocess path.
        _model = None
        _label_encoder = None
        return
    try:
        _model = joblib.load(str(_MODEL_PATH))
        _label_encoder = joblib.load(str(_ENCODER_PATH))
        logger.info("Accident prediction model loaded successfully")
    except FileNotFoundError as e:
        logger.warning(f"Accident model files not found: {e}")
    except Exception as e:
        logger.error(f"Failed to load accident model: {e}")


def is_loaded() -> bool:
    """Check whether model is ready."""
    if sys.version_info >= (3, 14):
        try:
            result = _run_subprocess_prediction(None)
            return bool(result.get("loaded", False))
        except Exception as exc:
            logger.error(f"Accident model subprocess status failed: {exc}")
            return False

    if _model is None or _label_encoder is None:
        load_model()
    return _model is not None and _label_encoder is not None


def predict(input_data: dict) -> dict:
    """
    Run accident severity prediction.

    Parameters
    ----------
    input_data : dict
        Keys: State, City, No_of_Vehicles, Road_Type, Road_Surface,
              Light_Condition, Weather, Casualty_Class, Casualty_Sex,
              Casualty_Age, Vehicle_Type

    Returns
    -------
    dict  with  prediction (str) and input_data echo
    """
    if sys.version_info >= (3, 14):
        try:
            result = _run_subprocess_prediction(input_data)
            return {
                "prediction": result.get("prediction"),
                "input_data": input_data,
            }
        except Exception as exc:
            logger.error(f"Accident prediction subprocess error: {exc}")
            return {"error": str(exc), "prediction": None}

    if not is_loaded():
        load_model()
    if not is_loaded():
        return {"error": "Model not loaded", "prediction": None}

    try:
        df = pd.DataFrame([input_data])
        raw_pred = _model.predict(df)
        severity = _label_encoder.inverse_transform(raw_pred)[0]
        return {
            "prediction": severity,
            "input_data": input_data,
        }
    except Exception as e:
        logger.error(f"Accident prediction error: {e}")
        return {"error": str(e), "prediction": None}
