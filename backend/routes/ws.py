"""
WebSocket Routes — Real-time risk data push to Dashboard.
"""

import asyncio
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services import drowsiness_service, fog_service, stress_service, visibility_service, kid_safety_service
from services.risk_engine import compute_unified_risk
from config import WEBSOCKET_PUSH_INTERVAL, FOG_POLL_INTERVAL, KID_SAFETY_POLL_INTERVAL
from utils.logger import get_logger

logger = get_logger("routes.ws")
router = APIRouter()

# Connected clients list
_clients: list[WebSocket] = []


@router.websocket("/ws/risk")
async def websocket_risk(ws: WebSocket):
    """
    Real-time risk data stream.
    Pushes unified risk JSON to the dashboard every WEBSOCKET_PUSH_INTERVAL.
    Also periodically runs fog detection on camera frames.
    """
    await ws.accept()
    _clients.append(ws)
    logger.info(f"WebSocket client connected ({len(_clients)} total)")

    try:
        while True:
            # Refresh all frame-based modules every tick so the dashboard turns active immediately.
            frame = drowsiness_service.get_frame()
            if frame:
                fog_service.predict(frame, user_id="system", image_name="ws_frame.jpg")
                visibility_service.predict(frame, user_id="system", image_name="ws_frame.jpg")
                kid_safety_service.predict(frame, user_id="system", image_name="ws_frame.jpg")

            # Compute unified risk
            d_state = drowsiness_service.get_state()
            f_state = fog_service.get_state()
            s_state = stress_service.estimate_from_context(d_state)
            v_state = visibility_service.get_state()
            k_state = kid_safety_service.get_state()
            risk = compute_unified_risk(d_state, f_state, s_state, v_state, k_state)
            risk["timestamp"] = time.time()

            try:
                await ws.send_json(risk)
            except Exception:
                break

            await asyncio.sleep(WEBSOCKET_PUSH_INTERVAL)
    except WebSocketDisconnect:
        pass
    finally:
        if ws in _clients:
            _clients.remove(ws)
        logger.info(f"WebSocket client disconnected ({len(_clients)} remaining)")
