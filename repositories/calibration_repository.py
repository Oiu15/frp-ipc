from __future__ import annotations

"""Filesystem-backed calibration repository.

Current scope is intentionally narrow:
- od_calibration.json
- id_calibration.json
- single-probe calibration companion json
- corresponding history jsonl paths

This is a staging repository. It keeps the legacy OD/ID active/history
formats compatible and introduces a dedicated single-probe calibration
json path without forcing app.py to migrate immediately.
"""

import json
from pathlib import Path
from typing import Any, Mapping

from application.contracts import CalibrationRepositoryProtocol
from application.state import CalibrationSnapshot


class CalibrationRepository(CalibrationRepositoryProtocol):
    """Repository facade for calibration json files and history paths."""

    def __init__(self, *, app_root_dir: Path | None = None) -> None:
        self._app_root_dir_override = Path(app_root_dir) if app_root_dir is not None else None

    def _app_root_dir(self) -> Path:
        try:
            if self._app_root_dir_override is not None:
                return self._app_root_dir_override
            return Path.home() / "FRP_IPC"
        except Exception:
            return Path("./FRP_IPC")

    def calibration_root_dir(self) -> Path:
        return self._app_root_dir() / "calibration"

    def od_calibration_file(self) -> Path:
        return self.calibration_root_dir() / "od_calibration.json"

    def od_history_file(self) -> Path:
        return self.calibration_root_dir() / "od_calibration_history.jsonl"

    def id_calibration_file(self) -> Path:
        return self.calibration_root_dir() / "id_calibration.json"

    def id_history_file(self) -> Path:
        return self.calibration_root_dir() / "id_calibration_history.jsonl"

    def id_single_calibration_file(self) -> Path:
        return self.calibration_root_dir() / "id_single_calibration.json"

    def id_single_history_file(self) -> Path:
        return self.calibration_root_dir() / "id_single_calibration_history.jsonl"

    def active_paths(self) -> dict[str, Path]:
        return {
            "od": self.od_calibration_file(),
            "id": self.id_calibration_file(),
            "id_single": self.id_single_calibration_file(),
        }

    def history_paths(self) -> dict[str, Path]:
        return {
            "od": self.od_history_file(),
            "id": self.id_history_file(),
            "id_single": self.id_single_history_file(),
        }

    def _as_mapping(self, value: Any) -> Mapping[str, Any]:
        return value if isinstance(value, Mapping) else {}

    def _load_json_file(self, path: Path) -> dict[str, Any]:
        try:
            if not path.exists():
                return {}
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            return dict(data) if isinstance(data, Mapping) else {}
        except Exception:
            return {}

    def _save_json_file(self, path: Path, data: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(data), f, ensure_ascii=False, indent=2)

    def _append_history(self, path: Path, data: Mapping[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(data), ensure_ascii=False) + "\n")
        except Exception:
            pass

    def load_od_active(self) -> dict[str, Any]:
        return self._load_json_file(self.od_calibration_file())

    def save_od_active(self, data: Mapping[str, Any]) -> None:
        payload = dict(data)
        self._save_json_file(self.od_calibration_file(), payload)
        self._append_history(self.od_history_file(), payload)

    def load_id_active(self) -> dict[str, Any]:
        return self._load_json_file(self.id_calibration_file())

    def save_id_active(self, data: Mapping[str, Any]) -> None:
        payload = dict(data)
        self._save_json_file(self.id_calibration_file(), payload)
        self._append_history(self.id_history_file(), payload)

    def load_id_single_active(self) -> dict[str, Any]:
        return self._load_json_file(self.id_single_calibration_file())

    def save_id_single_active(self, data: Mapping[str, Any]) -> None:
        payload = dict(data)
        self._save_json_file(self.id_single_calibration_file(), payload)
        self._append_history(self.id_single_history_file(), payload)

    def _as_float(self, value: Any, default: float | None = None) -> float | None:
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    def _as_bool(self, value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        s = str(value).strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
        return default

    def _pick(self, data: Mapping[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in data:
                return data[key]
        return None

    def _normalize_out1_map(self, value: Any) -> str:
        s = str(value or "L").strip().upper()
        return s if s in {"L", "R"} else "L"

    def _normalize_angle_src_ui(self, value: Any) -> str:
        s = str(value or "AX3").strip()
        return "无角度" if (("无" in s) or (s.upper() == "NONE")) else "AX3"

    def _normalize_template_mask(self, defects: Mapping[str, Any]) -> list[int]:
        tpl = defects.get("template_mask")
        if isinstance(tpl, list) and len(tpl) == 360:
            out: list[int] = []
            for item in tpl:
                try:
                    out.append(1 if int(item) else 0)
                except Exception:
                    out.append(0)
            return out

        ranges = defects.get("template_ranges")
        mask = [0] * 360
        if not isinstance(ranges, list):
            return mask

        for item in ranges:
            try:
                a = int(item[0]) % 360
                b = int(item[1]) % 360
            except Exception:
                continue

            if a <= b:
                for idx in range(a, b + 1):
                    mask[idx % 360] = 1
            else:
                for idx in range(a, 360):
                    mask[idx % 360] = 1
                for idx in range(0, b + 1):
                    mask[idx % 360] = 1
        return mask

    def load_od_prefill(self) -> dict[str, Any]:
        data = self.load_od_active()
        out_map = self._as_mapping(data.get("out_map"))
        params = self._as_mapping(data.get("params"))
        defects = self._as_mapping(data.get("defects"))
        return {
            "B_active": self._as_float(data.get("B_active")),
            "D_ref": self._as_float(data.get("D_ref")),
            "cmd_used": str(data.get("cmd_used", "") or ""),
            "out1_map": self._normalize_out1_map(out_map.get("OUT1", "L")),
            "angle_src_ui": self._normalize_angle_src_ui(params.get("angle_src")),
            "filter": str(params.get("filter", "无") or "无"),
            "outlier_sigma": str(params.get("outlier_sigma", "3.0") or "3.0"),
            "defect_template_mask": self._normalize_template_mask(defects),
        }

    def load_id_prefill(self) -> dict[str, Any]:
        data = self.load_id_active()
        return {
            "delta_c_mm": self._as_float(data.get("delta_c_mm")),
            "D_ref": self._as_float(data.get("D_ref")),
        }

    def load_id_single_prefill(self) -> dict[str, Any]:
        data = self.load_id_single_active()
        return {
            "id_single_enable": self._as_bool(self._pick(data, "id_single_enable", "enabled", "enable"), default=False),
            "id_single_k": self._as_float(self._pick(data, "id_single_k", "k", "K"), default=1.0),
            "id_single_b": self._as_float(self._pick(data, "id_single_b", "b", "B"), default=0.0),
            "D_ref": self._as_float(self._pick(data, "D_ref", "d_ref_mm")),
        }

    def load_snapshot(self) -> CalibrationSnapshot:
        od_data = self.load_od_active()
        id_data = self.load_id_active()
        id_single_data = self.load_id_single_active()

        out_map = od_data.get("out_map", {})
        if not isinstance(out_map, Mapping):
            out_map = {}

        id_single_enabled_raw = self._pick(id_single_data, "id_single_enable", "enabled", "enable")
        id_single_enabled = self._as_bool(id_single_enabled_raw, default=bool(id_single_data))

        id_single_k = self._as_float(
            self._pick(id_single_data, "id_single_k", "k", "K"),
            default=1.0,
        )
        if id_single_k is None:
            id_single_k = 1.0

        id_single_b = self._as_float(
            self._pick(id_single_data, "id_single_b", "b", "B"),
            default=0.0,
        )
        if id_single_b is None:
            id_single_b = 0.0

        return CalibrationSnapshot(
            od_b_active_mm=self._as_float(od_data.get("B_active"), default=0.0) or 0.0,
            od_out1_map=str(out_map.get("OUT1", "L") or "L").upper(),
            od_d_ref_mm=self._as_float(od_data.get("D_ref")),
            od_request_cmd=str(od_data.get("cmd_used", "") or ""),
            id_delta_c_mm=self._as_float(id_data.get("delta_c_mm"), default=0.0) or 0.0,
            id_d_ref_mm=self._as_float(id_data.get("D_ref")),
            id_single_enabled=id_single_enabled,
            id_single_k=float(id_single_k),
            id_single_b_mm=float(id_single_b),
            id_single_d_ref_mm=self._as_float(self._pick(id_single_data, "D_ref", "d_ref_mm")),
        )


__all__ = ["CalibrationRepository"]
