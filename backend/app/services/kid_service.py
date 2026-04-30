"""Facade for kid safety detection services."""

from services.kid_safety_service import get_state, load_model, predict


detect_kid = predict
