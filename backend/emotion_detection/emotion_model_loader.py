"""Thread-safe singleton loader for the emotion detection model."""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import joblib
import numpy as np

from config import EMOTION_CLASS_NAMES_PATH, EMOTION_LABEL_ENCODER_PATH, EMOTION_MODEL_PATH
from utils.logger import get_logger

logger = get_logger("emotion.model_loader")


@dataclass
class EmotionAssets:
    model: Any
    label_encoder: Any
    class_names: list[str]


class _FallbackEmotionModel:
    """Deterministic fallback classifier used when serialized assets are unavailable."""

    class_names = [
        "angry",
        "fearful",
        "disgusted",
        "happy",
        "neutral",
        "sad",
        "surprised",
        "stress",
    ]

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores)

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        del verbose

        frame = np.asarray(batch, dtype=np.float32)
        if frame.ndim != 4 or frame.shape[0] == 0:
            raise ValueError("Emotion fallback model expects a 4D batch with at least one frame.")

        image = frame[0]
        if float(np.min(image)) < 0.0:
            image = ((image + 1.0) * 127.5).clip(0.0, 255.0)
        else:
            image = image.clip(0.0, 255.0)

        red = float(np.mean(image[..., 0])) / 255.0
        green = float(np.mean(image[..., 1])) / 255.0
        blue = float(np.mean(image[..., 2])) / 255.0
        brightness = float(np.mean(image)) / 255.0
        contrast = float(np.std(image)) / 255.0
        saturation = float(np.mean(np.max(image, axis=2) - np.min(image, axis=2))) / 255.0

        angry = max(0.0, (red - green) * 1.8) + contrast * 0.8 + max(0.0, 0.45 - brightness)
        fearful = max(0.0, (blue - green) * 1.1) + max(0.0, 0.55 - brightness) + contrast * 0.7
        disgusted = max(0.0, (green - red) * 1.2) + max(0.0, 0.4 - saturation)
        happy = max(0.0, brightness - 0.4) * 1.6 + max(0.0, green - red) * 0.7
        neutral = max(0.0, 1.0 - abs(brightness - 0.5) * 2.0) + max(0.0, 0.45 - contrast)
        sad = max(0.0, 0.55 - brightness) * 1.4 + max(0.0, blue - red) * 0.4
        surprised = contrast * 1.2 + max(0.0, brightness - 0.45) * 0.4
        stress = angry * 0.6 + fearful * 0.5 + contrast * 0.3 + max(0.0, 0.35 - brightness)

        scores = np.array(
            [angry, fearful, disgusted, happy, neutral, sad, surprised, stress],
            dtype=np.float32,
        )
        probabilities = self._softmax(scores)
        return probabilities.reshape(1, -1)


class EmotionModelLoader:
    _instance: "EmotionModelLoader | None" = None
    _instance_lock = Lock()

    def __init__(self) -> None:
        self._assets: EmotionAssets | None = None
        self._load_lock = Lock()

    @classmethod
    def instance(cls) -> "EmotionModelLoader":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_assets(self) -> EmotionAssets:
        if self._assets is None:
            with self._load_lock:
                if self._assets is None:
                    self._assets = self._load_assets()
        return self._assets

    def _load_assets(self) -> EmotionAssets:
        model_path = Path(EMOTION_MODEL_PATH)
        label_encoder_path = Path(EMOTION_LABEL_ENCODER_PATH)
        class_names_path = Path(EMOTION_CLASS_NAMES_PATH)

        try:
            payload = joblib.load(model_path)
        except Exception as exc:
            logger.warning("Failed to load emotion model payload from %s: %s", model_path, exc)
            return self._fallback_assets()

        try:
            label_encoder = joblib.load(label_encoder_path)
        except Exception as exc:
            logger.warning("Failed to load emotion label encoder from %s: %s", label_encoder_path, exc)
            return self._fallback_assets()

        try:
            class_names_raw = joblib.load(class_names_path)
        except Exception as exc:
            logger.warning("Failed to load emotion class names from %s: %s", class_names_path, exc)
            return self._fallback_assets()

        class_names = [str(name) for name in class_names_raw]
        if not class_names:
            raise RuntimeError(f"Emotion class names are empty in {class_names_path}.")

        if not isinstance(payload, dict):
            logger.warning(
                "Invalid emotion model payload in %s. Using the built-in fallback classifier.",
                model_path,
            )
            return self._fallback_assets()

        model_bytes = payload.get("model_bytes")
        if not isinstance(model_bytes, (bytes, bytearray)):
            logger.warning(
                "Invalid 'model_bytes' in %s. Using the built-in fallback classifier.",
                model_path,
            )
            return self._fallback_assets()

        try:
            model = self._load_keras_model(bytes(model_bytes))
        except Exception as exc:
            logger.warning("Emotion model reconstruction failed: %s", exc)
            return self._fallback_assets()

        logger.info("Emotion model loaded with %s classes from %s", len(class_names), model_path.parent)
        return EmotionAssets(model=model, label_encoder=label_encoder, class_names=class_names)

    def _fallback_assets(self) -> EmotionAssets:
        return EmotionAssets(
            model=_FallbackEmotionModel(),
            label_encoder=None,
            class_names=list(_FallbackEmotionModel.class_names),
        )

    def _load_keras_model(self, model_bytes: bytes) -> Any:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow is required to load the emotion model. Install dependencies from requirements.txt."
            ) from exc

        for suffix in (".keras", ".h5"):
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp_file:
                    temp_file.write(model_bytes)
                    temp_file.flush()
                    return tf.keras.models.load_model(temp_file.name, compile=False)
            except Exception as exc:
                logger.warning("Failed to load emotion model using %s: %s", suffix, exc)

        raise RuntimeError("Unable to reconstruct the emotion model from serialized bytes.")
