from __future__ import annotations

"""Filesystem-backed system settings repository."""

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Mapping

from core.serial_service import default_serial_template, normalize_serial_template, validate_serial_template

logger = logging.getLogger("frp.settings")


class SettingsRepository:
    """Load and save system settings outside recipe storage."""

    def __init__(self, *, app_root_dir: Path | None = None) -> None:
        self._app_root_dir_override = Path(app_root_dir) if app_root_dir is not None else None

    def _app_root_dir(self) -> Path:
        try:
            if self._app_root_dir_override is not None:
                return self._app_root_dir_override
            return Path.home() / "FRP_IPC"
        except Exception:
            return Path("./FRP_IPC")

    def settings_path(self) -> Path:
        return self._app_root_dir() / "settings.json"

    def load_settings(self) -> dict[str, Any]:
        path = self.settings_path()
        if not path.exists():
            return self.default_settings()
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return self.normalize_settings(payload)
        except Exception as exc:
            self._backup_corrupt_settings(path, exc)
            return self.default_settings()

    def save_settings(self, settings: Mapping[str, Any]) -> None:
        normalized = self.normalize_settings(settings)
        path = self.settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)
        tmp.replace(path)

    def load_serial_template(self) -> dict[str, Any]:
        return dict(self.load_settings()["serial_template"])

    def save_serial_template(self, template: Mapping[str, Any]) -> None:
        settings = self.load_settings()
        settings["serial_template"] = normalize_serial_template(template)
        validate_serial_template(settings["serial_template"])
        self.save_settings(settings)

    @staticmethod
    def default_settings() -> dict[str, Any]:
        return {"serial_template": default_serial_template()}

    @staticmethod
    def normalize_settings(settings: Mapping[str, Any] | None) -> dict[str, Any]:
        if not isinstance(settings, Mapping):
            raise ValueError("settings must be an object")
        raw_template = settings.get("serial_template", default_serial_template())
        template = normalize_serial_template(raw_template)
        validate_serial_template(template)
        return {"serial_template": template}

    def _backup_corrupt_settings(self, path: Path, exc: Exception) -> None:
        try:
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            backup = path.with_name(f"settings.corrupt-{stamp}.json")
            path.replace(backup)
            logger.warning("Backed up corrupt settings file to %s: %s", backup, exc)
        except Exception:
            logger.exception("Failed to back up corrupt settings file %s", path)


__all__ = ["SettingsRepository"]
