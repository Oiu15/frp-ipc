# ./utils/logger.py
from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

_inited = False
_app_log_path: Optional[Path] = None


class _PrefixFilter(logging.Filter):
    def __init__(self, prefixes: Iterable[str]):
        super().__init__()
        self._prefixes = tuple(str(x).strip() for x in prefixes if str(x).strip())

    def filter(self, record: logging.LogRecord) -> bool:
        name = str(getattr(record, "name", "") or "")
        for p in self._prefixes:
            if name == p or name.startswith(p + "."):
                return True
        return False


def _app_dir() -> Path:
    if getattr(sys, "frozen", False):
        try:
            return Path(sys.executable).resolve().parent
        except Exception:
            return Path(".").resolve()
    try:
        return Path(__file__).resolve().parents[1]
    except Exception:
        return Path(".").resolve()


def _resolve_log_dir(filename: str, log_dir: Optional[str]) -> Path:
    if log_dir:
        try:
            return Path(log_dir).expanduser().resolve()
        except Exception:
            return (_app_dir() / "logs").resolve()

    try:
        fp = Path(filename)
    except Exception:
        return (_app_dir() / "logs").resolve()

    if fp.is_absolute():
        return fp.parent

    # Keep backward compatibility with relative filenames under app root.
    rel = _app_dir() / fp
    if fp.parent != Path("."):
        return rel.parent

    # Plain filename (e.g. "log.txt"): place standard logs under ./logs
    return (_app_dir() / "logs").resolve()


def _clear_root_handlers(root: logging.Logger) -> None:
    for h in list(root.handlers):
        try:
            root.removeHandler(h)
        except Exception:
            pass
        try:
            h.close()
        except Exception:
            pass


def init_log(
    filename: str = "log.txt",
    overwrite: bool = False,
    *,
    log_dir: Optional[str] = None,
) -> Path:
    """Initialize standard logging handlers for FRP IPC.

    Outputs:
    - app.log     : all records, level>=DEBUG
    - comm.log    : frp.plc/frp.modbus/frp.gauge, level>=DEBUG
    - measure.log : frp.autoflow/frp.algo/frp.data, level>=DEBUG
    - console     : all records, level>=INFO
    """
    global _inited, _app_log_path
    if _inited and _app_log_path is not None:
        return _app_log_path

    root_dir = _resolve_log_dir(filename=filename, log_dir=log_dir)
    try:
        root_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        root_dir = (_app_dir() / "logs").resolve()
        try:
            root_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    app_path = root_dir / "app.log"
    comm_path = root_dir / "comm.log"
    measure_path = root_dir / "measure.log"

    root = logging.getLogger()
    _clear_root_handlers(root)
    root.setLevel(logging.DEBUG)

    file_fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s] [%(threadName)s] %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    app_handler = TimedRotatingFileHandler(
        app_path,
        when="midnight",
        interval=1,
        backupCount=14,
        encoding="utf-8",
        utc=False,
    )
    app_handler.setLevel(logging.DEBUG)
    app_handler.setFormatter(file_fmt)

    comm_handler = TimedRotatingFileHandler(
        comm_path,
        when="midnight",
        interval=1,
        backupCount=14,
        encoding="utf-8",
        utc=False,
    )
    comm_handler.setLevel(logging.DEBUG)
    comm_handler.setFormatter(file_fmt)
    comm_handler.addFilter(_PrefixFilter(("frp.plc", "frp.modbus", "frp.gauge")))

    measure_handler = TimedRotatingFileHandler(
        measure_path,
        when="midnight",
        interval=1,
        backupCount=14,
        encoding="utf-8",
        utc=False,
    )
    measure_handler.setLevel(logging.DEBUG)
    measure_handler.setFormatter(file_fmt)
    measure_handler.addFilter(_PrefixFilter(("frp.autoflow", "frp.algo", "frp.data")))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_fmt)

    root.addHandler(app_handler)
    root.addHandler(comm_handler)
    root.addHandler(measure_handler)
    root.addHandler(console_handler)

    _inited = True
    _app_log_path = app_path
    return app_path


def _fmt(v: Any) -> str:
    try:
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)
    except Exception:
        return "<fmt_err>"


def log(msg: str, **fields: Any) -> None:
    """Backward-compatible legacy logging helper."""
    try:
        if not _inited:
            init_log(overwrite=False)
    except Exception:
        return

    try:
        logger = logging.getLogger("frp.app")
        event = str(msg)
        if fields:
            parts = [f"{k}={_fmt(v)}" for k, v in fields.items()]
            event = f"{event} " + " ".join(parts)
        logger.info(event)
    except Exception:
        return


def log_exc(msg: str, exc: Optional[BaseException] = None) -> None:
    """Backward-compatible legacy exception logger."""
    try:
        if not _inited:
            init_log(overwrite=False)
    except Exception:
        return

    try:
        logger = logging.getLogger("frp.app")
        if exc is None:
            logger.exception(str(msg))
        else:
            logger.exception("%s | exc=%s", str(msg), _fmt(exc))
    except Exception:
        return
