"""Audio alert service with one-shot and stop-aware looping playback."""

from __future__ import annotations

import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger("audio_alert_service")

try:
    from playsound import playsound
except Exception:
    playsound = None

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_ALERT_CANDIDATES: Dict[str, List[Path]] = {
    "drowsiness": [
        _PROJECT_ROOT / "alert.wav",
        _PROJECT_ROOT / "Alert.wav",
        _PROJECT_ROOT / "Drowsiness_and_Yawning_Detection" / "alert.wav",
        _PROJECT_ROOT / "Drowsiness_and_Yawning_Detection" / "Alert.wav",
    ],
    "focus": [
        _PROJECT_ROOT / "alert.wav",
        _PROJECT_ROOT / "Alert.wav",
        _PROJECT_ROOT / "Drowsiness_and_Yawning_Detection" / "alert.wav",
        _PROJECT_ROOT / "Drowsiness_and_Yawning_Detection" / "Alert.wav",
    ],
    "yawning": [
        _PROJECT_ROOT / "alert2.wav",
        _PROJECT_ROOT / "Alert2.wav",
    ],
    "stress": [
        _PROJECT_ROOT / "alert3.wav",
        _PROJECT_ROOT / "Alert3.wav",
    ],
}

_lock = threading.Lock()
_last_play_ts: Dict[str, float] = {}
_missing_warned: set[str] = set()
_loop_controllers: Dict[str, dict] = {}


def _resolve_alert_file(alert_key: str) -> Optional[Path]:
    for candidate in _ALERT_CANDIDATES.get(alert_key, []):
        if candidate.is_file():
            return candidate

    if alert_key not in _missing_warned:
        logger.warning(
            "Audio file not found for '%s'. Expected one of: %s",
            alert_key,
            ", ".join(str(p) for p in _ALERT_CANDIDATES.get(alert_key, [])),
        )
        _missing_warned.add(alert_key)
    return None


def _player_command(path: Path) -> Optional[list[str]]:
    system = platform.system().lower()

    if system == "darwin" and shutil.which("afplay"):
        return ["afplay", str(path)]

    if system == "linux" and shutil.which("aplay"):
        return ["aplay", str(path)]

    return None


def _terminate_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return

    try:
        process.terminate()
        process.wait(timeout=1.0)
    except Exception:
        try:
            process.kill()
            process.wait(timeout=1.0)
        except Exception:
            pass


def _play_with_fallback(path: Path) -> None:
    command = _player_command(path)
    if command is not None:
        subprocess.run(command, check=False)
        return

    if playsound is not None:
        playsound(str(path))
        return

    system = platform.system().lower()
    if system == "windows":
        import winsound

        winsound.PlaySound(str(path), winsound.SND_FILENAME)
        return

    raise RuntimeError(f"No supported audio player found for {path}")


def _loop_worker(alert_key: str, path: Path, stop_event: threading.Event) -> None:
    system = platform.system().lower()

    if system == "windows":
        import winsound

        try:
            winsound.PlaySound(
                str(path),
                winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP,
            )
            stop_event.wait()
        finally:
            winsound.PlaySound(None, 0)
            with _lock:
                _loop_controllers.pop(alert_key, None)
        return

    command = _player_command(path)
    if command is None and playsound is None:
        logger.warning("No looping audio backend available for '%s'", alert_key)
        with _lock:
            _loop_controllers.pop(alert_key, None)
        return

    process: subprocess.Popen | None = None

    try:
        while not stop_event.is_set():
            if command is not None:
                process = subprocess.Popen(command)
                with _lock:
                    controller = _loop_controllers.get(alert_key)
                    if controller is not None:
                        controller["process"] = process

                while process.poll() is None:
                    if stop_event.wait(0.1):
                        _terminate_process(process)
                        break
            else:
                _play_with_fallback(path)
                if stop_event.wait(0.05):
                    break
    except Exception as exc:
        logger.warning("Failed to loop '%s' alert from %s: %s", alert_key, path, exc)
    finally:
        _terminate_process(process)
        with _lock:
            _loop_controllers.pop(alert_key, None)
        logger.info("Audio alert loop stopped: %s", alert_key)


def _worker(alert_key: str, path: Path) -> None:
    try:
        _play_with_fallback(path)
        logger.info("Audio alert played: %s (%s)", alert_key, path.name)
    except Exception as exc:
        logger.warning("Failed to play '%s' alert from %s: %s", alert_key, path, exc)


def start_alert_loop(alert_key: str) -> bool:
    """Continuously play an alert until stop_alert() is called."""
    path = _resolve_alert_file(alert_key)
    if path is None:
        return False

    with _lock:
        controller = _loop_controllers.get(alert_key)
        if controller and controller["thread"].is_alive():
            return False

        stop_event = threading.Event()
        thread = threading.Thread(
            target=_loop_worker,
            args=(alert_key, path, stop_event),
            daemon=True,
        )
        _loop_controllers[alert_key] = {
            "stop_event": stop_event,
            "thread": thread,
            "process": None,
        }
        thread.start()

    logger.info("Audio alert loop started: %s", alert_key)
    return True


def stop_alert(alert_key: str) -> bool:
    """Stop a looping alert if it is currently active."""
    with _lock:
        controller = _loop_controllers.get(alert_key)
        if controller is None:
            return False

        controller["stop_event"].set()
        process = controller.get("process")

    _terminate_process(process)
    return True


def trigger_alert(alert_key: str, cooldown_seconds: float = 4.0) -> bool:
    """Trigger alert sound once in a background thread, honoring cooldown per type."""
    path = _resolve_alert_file(alert_key)
    if path is None:
        return False

    now = time.monotonic()
    with _lock:
        last = _last_play_ts.get(alert_key, 0.0)
        if now - last < cooldown_seconds:
            return False
        _last_play_ts[alert_key] = now

    thread = threading.Thread(target=_worker, args=(alert_key, path), daemon=True)
    thread.start()
    return True
