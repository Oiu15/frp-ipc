from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from config.addresses import (
    DEFAULT_GAUGE_PORT,
    DEFAULT_PLC_IP,
    DEFAULT_PLC_PORT,
    DEFAULT_UNIT_ID,
    FLOAT64_WORD_ORDER,
    POLL_INTERVAL_S,
)


def default_app_root_dir() -> Path:
    try:
        return Path.home() / "FRP_IPC"
    except Exception:
        return Path("./FRP_IPC")


@dataclass(frozen=True, slots=True)
class PlcConfig:
    ip: str = DEFAULT_PLC_IP
    port: int = DEFAULT_PLC_PORT
    unit_id: int = DEFAULT_UNIT_ID
    poll_interval_s: float = POLL_INTERVAL_S
    word_order: str = FLOAT64_WORD_ORDER


@dataclass(frozen=True, slots=True)
class GaugeConfig:
    port: str = DEFAULT_GAUGE_PORT
    baud: int = 115200
    timeout_s: float = 0.5
    eol: str = "\r"
    request_cmd: str = "M1,1"
    bytesize: int = 8
    parity: str = "N"
    stopbits: int = 1


@dataclass(frozen=True, slots=True)
class PathConfig:
    app_root_dir: Path = field(default_factory=default_app_root_dir)
    recipe_profile_name: str = "FRP_IPC"
    fallback_recipe_dir: Path = field(default_factory=lambda: Path("./data/recipes"))


@dataclass(frozen=True, slots=True)
class AppConfig:
    plc: PlcConfig = field(default_factory=PlcConfig)
    gauge: GaugeConfig = field(default_factory=GaugeConfig)
    paths: PathConfig = field(default_factory=PathConfig)


def default_app_config(app_root_dir: Path | str | None = None) -> AppConfig:
    paths = PathConfig(app_root_dir=Path(app_root_dir) if app_root_dir is not None else default_app_root_dir())
    return AppConfig(paths=paths)


__all__ = ["AppConfig", "GaugeConfig", "PathConfig", "PlcConfig", "default_app_config", "default_app_root_dir"]
