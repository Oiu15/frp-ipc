from __future__ import annotations

"""Run identity helper mixin for AppHost.

Extracted from ``app_host.py`` to reduce monolith size.
Provides: serial number generation, run counter persistence, device code.
"""

import datetime
import json
import platform
import re
from pathlib import Path
from typing import Optional


class HostIdentityMixin:
    """Mixin providing serial-number / run-identity helpers.

    Requires the main class to set ``self._run_session`` before any
    property accessor is called.
    """

    # -- run-session property delegation ----------------------------------

    @property
    def _run_serial(self) -> Optional[str]:
        return self._run_session.serial

    @_run_serial.setter
    def _run_serial(self, value: Optional[str]) -> None:
        self._run_session.serial = value

    @property
    def _run_id(self) -> Optional[str]:
        return self._run_session.run_id

    @_run_id.setter
    def _run_id(self, value: Optional[str]) -> None:
        self._run_session.run_id = value

    @property
    def _run_start_ts(self) -> Optional[float]:
        return self._run_session.start_ts

    @_run_start_ts.setter
    def _run_start_ts(self, value: Optional[float]) -> None:
        self._run_session.start_ts = value

    @property
    def _run_end_ts(self) -> Optional[float]:
        return self._run_session.end_ts

    @_run_end_ts.setter
    def _run_end_ts(self, value: Optional[float]) -> None:
        self._run_session.end_ts = value

    # -- filesystem helpers ----------------------------------------------

    def _sanitize_recipe_key(self, name: str) -> str:
        """Recipe key used in serial/filenames (keep readable but filesystem-safe)."""
        s = str(name or "").strip()
        if not s:
            s = "recipe"
        s2: list[str] = []
        for ch in s:
            o = ord(ch)
            if ch.isalnum() or ch in "_-":
                s2.append(ch)
            elif 0x4E00 <= o <= 0x9FFF:  # CJK Unified Ideographs
                s2.append(ch)
            else:
                s2.append("_")
        out = "".join(s2)
        out = re.sub(r"_+", "_", out).strip("_")
        return out[:24] if out else "recipe"

    def _app_root_dir(self) -> Path:
        try:
            return Path.home() / "FRP_IPC"
        except Exception:
            return Path("./FRP_IPC")

    def _counter_file(self) -> Path:
        return self._app_root_dir() / "run_counter.json"

    # -- run counter persistence -----------------------------------------

    def _load_run_counters(self) -> dict:
        p = self._counter_file()
        try:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f) or {}
        except Exception:
            pass
        return {}

    def _save_run_counters(self, data: dict) -> None:
        p = self._counter_file()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _next_serial(self, recipe_name: str) -> str:
        today = datetime.date.today()
        day_tag = today.strftime("%Y%m%d")
        recipe_key = self._sanitize_recipe_key(recipe_name)
        counters = self._load_run_counters()
        day_map = counters.get(day_tag, {})
        try:
            seq = int(day_map.get(recipe_key, 0)) + 1
        except Exception:
            seq = 1
        day_map[recipe_key] = seq
        counters[day_tag] = day_map
        self._save_run_counters(counters)
        return f"{day_tag}-{recipe_key}-{seq:03d}"

    def _get_device_code(self) -> str:
        """Best-effort stable device code (used in export meta)."""
        # Prefer Windows MachineGuid when available.
        try:
            import winreg  # type: ignore
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography")
            v, _t = winreg.QueryValueEx(k, "MachineGuid")
            if v:
                return str(v)
        except Exception:
            pass
        # Fallback: hostname + MAC
        try:
            import uuid as _uuid
            mac = _uuid.getnode()
            return f"{platform.node()}-{mac:012x}"
        except Exception:
            return platform.node()
