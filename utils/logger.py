# ./utils/logger.py
from __future__ import annotations

"""Lightweight file logger.

- Writes plain text lines to log.txt in the application directory.
- Safe-by-design: never raises to caller.
- Works in script mode and (best-effort) in frozen mode.

Usage:
    from utils.logger import init_log, log, log_exc
    init_log()  # once at app start
    log("EVENT", key=value, ...)
"""

import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

_lock = threading.Lock()
_log_path: Optional[Path] = None
_inited = False


def _app_dir() -> Path:
    # If frozen (PyInstaller), write next to the executable.
    if getattr(sys, "frozen", False):
        try:
            return Path(sys.executable).resolve().parent
        except Exception:
            return Path(".").resolve()
    # Otherwise, write next to app.py (one level above this utils folder).
    try:
        return Path(__file__).resolve().parents[1]
    except Exception:
        return Path(".").resolve()


def init_log(filename: str = "log.txt", overwrite: bool = True) -> Path:
    """Initialize the logger and create/truncate log file.

    If `filename` is an absolute path, it will be used directly.
    Otherwise it is treated as relative to the application directory.
    """
    global _log_path, _inited
    try:
        fp = Path(filename)
        if fp.is_absolute():
            p = fp
        else:
            p = _app_dir() / filename
    except Exception:
        p = _app_dir() / filename
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    try:
        mode = "w" if overwrite else "a"
        with open(p, mode, encoding="utf-8") as f:
            f.write(f"# log start {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    except Exception:
        # Fall back to current working directory
        p = Path(".").resolve() / filename
        try:
            mode = "w" if overwrite else "a"
            with open(p, mode, encoding="utf-8") as f:
                f.write(f"# log start {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception:
            p = None  # type: ignore

    _log_path = p
    _inited = True
    return p  # type: ignore


def _fmt(v: Any) -> str:
    try:
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)
    except Exception:
        return "<fmt_err>"


def log(event: str, **fields: Any) -> None:
    """Append one line to log.txt."""
    global _log_path
    if not _inited:
        try:
            init_log(overwrite=False)
        except Exception:
            return

    p = _log_path
    if p is None:
        return

    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        th = threading.current_thread().name
        parts = [f"{k}={_fmt(v)}" for k, v in fields.items()]
        line = f"{ts} [{th}] {event}"
        if parts:
            line += " " + " ".join(parts)
        line += "\n"

        with _lock:
            with open(p, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        return


def log_exc(event: str, exc: BaseException) -> None:
    """Log an exception + traceback."""
    try:
        log(event, exc=str(exc))
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        for ln in tb.splitlines():
            log("TRACE", line=ln)
    except Exception:
        return
