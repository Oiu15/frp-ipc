from __future__ import annotations

"""Run export repository copied from the legacy app export flow.

This module intentionally keeps the legacy directory structure, CSV schema,
and field naming unchanged so the first migration step stays behavior-close
to the current app.py implementation.
"""

import csv
import datetime
import json
import platform
import re
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from application.contracts import RunRepositoryProtocol
from application.state import RunContext, RunIdentity
from core.models import Recipe


class RunRepository(RunRepositoryProtocol):
    """Filesystem-backed repository for run identity allocation and exports."""

    def __init__(
        self,
        *,
        app_root_dir: Path | None = None,
        software_version: str = "",
        device_code: str | None = None,
        plc_info: Mapping[str, Any] | None = None,
        gauge_info: Mapping[str, Any] | None = None,
    ) -> None:
        self._app_root_dir_override = Path(app_root_dir) if app_root_dir is not None else None
        self._software_version = str(software_version or "")
        self._device_code_override = str(device_code) if device_code else None
        self._plc_info = dict(plc_info or {})
        self._gauge_info = dict(gauge_info or {})

    def _app_root_dir(self) -> Path:
        try:
            if self._app_root_dir_override is not None:
                return self._app_root_dir_override
            return Path.home() / "FRP_IPC"
        except Exception:
            return Path("./FRP_IPC")

    def _exports_root_dir(self) -> Path:
        return self._app_root_dir() / "exports"

    def _counter_file(self) -> Path:
        return self._app_root_dir() / "run_counter.json"

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

    def _sanitize_recipe_key(self, name: str) -> str:
        s = str(name or "").strip()
        if not s:
            s = "recipe"
        s2: list[str] = []
        for ch in s:
            o = ord(ch)
            if ch.isalnum() or ch in "_-":
                s2.append(ch)
            elif 0x4E00 <= o <= 0x9FFF:
                s2.append(ch)
            else:
                s2.append("_")
        out = "".join(s2)
        out = re.sub(r"_+", "_", out).strip("_")
        return out[:24] if out else "recipe"

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

    def prepare_run(self, recipe_name: str) -> RunIdentity:
        serial = self._next_serial(recipe_name)
        return RunIdentity(
            serial=serial,
            run_id=str(uuid.uuid4()),
            started_at_ts=float(time.time()),
        )

    def _get_device_code(self) -> str:
        if self._device_code_override:
            return self._device_code_override
        try:
            import winreg  # type: ignore

            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography")
            v, _t = winreg.QueryValueEx(k, "MachineGuid")
            if v:
                return str(v)
        except Exception:
            pass
        try:
            import uuid as _uuid

            mac = _uuid.getnode()
            return f"{platform.node()}-{mac:012x}"
        except Exception:
            return platform.node()

    def _recipe_dump_dict(self, r: Recipe) -> dict:
        return {
            "name": r.name,
            "pipe_len_mm": r.pipe_len_mm,
            "clamp_occupy_mm": r.clamp_occupy_mm,
            "margin_head_mm": r.margin_head_mm,
            "margin_tail_mm": r.margin_tail_mm,
            "meas_total_len_mm": float(getattr(r, "meas_total_len_mm", 0.0) or 0.0),
            "section_count": r.section_count,
            "scan_axis": r.scan_axis,
            "scan_mode": str(getattr(r, "scan_mode", "sync") or "sync"),
            "disable_id_modbus": bool(getattr(r, "disable_id_modbus", False)),
            "split_keep_spinning": bool(getattr(r, "split_keep_spinning", True)),
            "split_slip_check": bool(getattr(r, "split_slip_check", True)),
            "split_slip_max_deg": float(getattr(r, "split_slip_max_deg", 5.0) or 5.0),
            "split_omega_cv_max": float(getattr(r, "split_omega_cv_max", 0.25) or 0.25),
            "teach_axes_mode": int(getattr(r, "teach_axes_mode", 2)),
            "od_std_mm": r.od_std_mm,
            "id_std_mm": r.id_std_mm,
            "od_tol_mm": r.od_tol_mm,
            "points_per_rev": r.points_per_rev,
            "sample_coverage": r.min_bin_coverage,
            "section_timeout_s": r.sample_timeout_s,
            "max_revs": r.max_revolutions,
            "rot_vel_velmove": float(getattr(r, "rot_vel_velmove", 200.0) or 200.0),
            "fit_strategy": str(getattr(r, "fit_strategy", "b 原始点按bin权重均衡")),
            "calc_input_mode": str(getattr(r, "calc_input_mode", "bin")),
            "bin_count": int(getattr(r, "bin_count", 90)),
            "bin_method": str(getattr(r, "bin_method", "median")),
            "pp_mode": str(getattr(r, "pp_mode", "p99_p1")),
            "theta_delay_s": float(getattr(r, "theta_delay_s", 0.0) or 0.0),
            "od_use_edges": bool(getattr(r, "od_use_edges", False)),
            "id_use_fit": bool(getattr(r, "id_use_fit", False)),
            "id_single_enable": bool(getattr(r, "id_single_enable", False)),
            "id_single_k": float(getattr(r, "id_single_k", 1.0) or 1.0),
            "id_single_b": float(getattr(r, "id_single_b", 0.0) or 0.0),
            "id_single_show_debug": bool(getattr(r, "id_single_show_debug", False)),
            "len_enable": bool(getattr(r, "len_enable", False)),
            "len_low_approach_abs": float(getattr(r, "len_low_approach_abs", 0.0) or 0.0),
            "len_low_search_dist": float(getattr(r, "len_low_search_dist", 220.0)),
            "len_high_search_dist": float(getattr(r, "len_high_search_dist", 220.0)),
            "len_search_vel": float(getattr(r, "len_search_vel", 5.0)),
            "len_search_timeout_s": float(getattr(r, "len_search_timeout_s", 12.0)),
            "len_tol_mm": float(getattr(r, "len_tol_mm", 20.0)),
            "len_high_margin": float(getattr(r, "len_high_margin", 20.0)),
            "len_debounce_k": int(getattr(r, "len_debounce_k", 6)),
            "len_max_stale_ms": int(getattr(r, "len_max_stale_ms", 300)),
            "len_backoff_mm": float(getattr(r, "len_backoff_mm", 2.0)),
            "section_pos_z": getattr(r, "section_pos_z", []),
            "standby_valid": bool(getattr(r, "standby_valid", False)),
            "standby_ax0_abs": float(getattr(r, "standby_ax0_abs", 0.0)),
            "standby_ax1_abs": float(getattr(r, "standby_ax1_abs", 0.0)),
            "standby_ax4_abs": float(getattr(r, "standby_ax4_abs", 0.0)),
            "start_valid": bool(getattr(r, "start_valid", False)),
            "start_ax0_abs": float(getattr(r, "start_ax0_abs", 0.0)),
            "ax2_len_valid": bool(getattr(r, "ax2_len_valid", False)),
            "ax2_len_abs": float(getattr(r, "ax2_len_abs", 0.0)),
            "ax2_rot_valid": bool(getattr(r, "ax2_rot_valid", False)),
            "ax2_rot_abs": float(getattr(r, "ax2_rot_abs", 0.0)),
        }

    def _cov_reason_text(self, reason: str) -> str:
        r = str(reason or "").strip()
        if not r:
            return ""
        mapping = {
            "COV": "覆盖率达标",
            "TIMEOUT": "超时退出",
            "REV": "圈数到达",
        }
        return mapping.get(r.upper(), r)

    def _format_cov_cols(self, info: Mapping[str, Any]) -> tuple[str, str, str, str, str, str]:
        cov = info.get("cov", None)
        if cov is None:
            return ("--", "--", "--", "--", "--", "")
        try:
            cov_pct = f"{float(cov) * 100:.1f}"
        except Exception:
            cov_pct = "--"
        miss = info.get("miss", None)
        try:
            miss_bin = "--" if miss is None else str(int(miss))
        except Exception:
            miss_bin = "--"
        max_gap = info.get("max_gap_deg", None)
        try:
            max_gap_deg = "--" if max_gap is None else f"{float(max_gap):.1f}"
        except Exception:
            max_gap_deg = "--"
        revs = info.get("revs", None)
        try:
            revs_txt = "--" if revs is None else f"{float(revs):.2f}"
        except Exception:
            revs_txt = "--"
        elapsed = info.get("elapsed", None)
        try:
            elapsed_s = "--" if elapsed is None else f"{float(elapsed):.2f}"
        except Exception:
            elapsed_s = "--"
        reason_txt = self._cov_reason_text(str(info.get("reason", "") or ""))
        return (cov_pct, miss_bin, max_gap_deg, revs_txt, elapsed_s, reason_txt)

    def export_run(self, context: RunContext) -> str:
        start_ts = float(context.identity.started_at_ts)
        end_ts = float(context.finished_at_ts if context.finished_at_ts is not None else time.time())
        serial = str(context.identity.serial)
        run_id = str(context.identity.run_id)

        day_dir = self._exports_root_dir() / datetime.date.fromtimestamp(start_ts).strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)

        def _safe_float(v: Any) -> float | str:
            if v is None:
                return ""
            try:
                return float(v)
            except Exception:
                return ""

        run_dir = day_dir / serial
        run_dir.mkdir(parents=True, exist_ok=True)

        section_csv = run_dir / "section_results.csv"
        rows = list(context.rows or [])
        section_coverage = dict(context.section_coverage or {})
        with open(section_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "serial", "run_id",
                "start_time", "end_time", "duration_s",
                "section_idx", "z_pos_mm",
                "od_avg_mm", "od_dev_mm", "od_runout_mm", "od_round_mm", "od_e_mm", "od_phi_deg",
                "id_avg_mm", "id_dev_mm", "id_runout_mm", "id_round_mm", "id_e_mm", "id_phi_deg",
                "concentricity_mm", "split_shift_deg", "coax_unreliable", "od_ecc_mm", "id_ecc_mm",
                "cov_pct", "miss_bin", "max_gap_deg", "revs", "cov_elapsed_s", "cov_reason",
                "od_round_fit_mm", "od_round_fit_rob_mm",
                "id_round_fit_mm", "id_round_fit_rob_mm",
                "od_pp_mm", "od_pp_rob_mm", "id_pp_mm", "id_pp_rob_mm",
                "raw",
            ])
            for r in rows:
                cov_info = section_coverage.get(int(getattr(r, "idx", 0) or 0), {})
                cov_cols = self._format_cov_cols(cov_info)
                w.writerow([
                    serial, run_id,
                    datetime.datetime.fromtimestamp(start_ts).isoformat(sep=" ", timespec="seconds"),
                    datetime.datetime.fromtimestamp(end_ts).isoformat(sep=" ", timespec="seconds"),
                    f"{(end_ts - start_ts):.3f}",
                    int(getattr(r, "idx", 0)),
                    _safe_float(getattr(r, "x_ui", 0.0)),
                    _safe_float(getattr(r, "od_avg", 0.0)),
                    _safe_float(getattr(r, "od_dev", 0.0)),
                    _safe_float(getattr(r, "od_runout", 0.0)),
                    _safe_float(getattr(r, "od_round", 0.0)),
                    "" if getattr(r, "od_e", None) is None else float(getattr(r, "od_e", 0.0)),
                    "" if getattr(r, "od_phi_deg", None) is None else float(getattr(r, "od_phi_deg", 0.0)),
                    _safe_float(getattr(r, "id_avg", 0.0)),
                    _safe_float(getattr(r, "id_dev", 0.0)),
                    _safe_float(getattr(r, "id_runout", 0.0)),
                    _safe_float(getattr(r, "id_round", 0.0)),
                    "" if getattr(r, "id_e", None) is None else float(getattr(r, "id_e", 0.0)),
                    "" if getattr(r, "id_phi_deg", None) is None else float(getattr(r, "id_phi_deg", 0.0)),
                    _safe_float(getattr(r, "concentricity", 0.0)),
                    "" if getattr(r, "split_shift_deg", None) is None else float(getattr(r, "split_shift_deg", 0.0)),
                    "" if getattr(r, "coax_unreliable", None) is None else (1 if bool(getattr(r, "coax_unreliable")) else 0),
                    "" if getattr(r, "od_ecc", None) is None else float(getattr(r, "od_ecc", 0.0)),
                    "" if getattr(r, "id_ecc", None) is None else float(getattr(r, "id_ecc", 0.0)),
                    *cov_cols,
                    "" if getattr(r, "od_round_fit_mm", None) is None else float(getattr(r, "od_round_fit_mm", 0.0)),
                    "" if getattr(r, "od_round_fit_rob_mm", None) is None else float(getattr(r, "od_round_fit_rob_mm", 0.0)),
                    "" if getattr(r, "id_round_fit_mm", None) is None else float(getattr(r, "id_round_fit_mm", 0.0)),
                    "" if getattr(r, "id_round_fit_rob_mm", None) is None else float(getattr(r, "id_round_fit_rob_mm", 0.0)),
                    "" if getattr(r, "od_pp_mm", None) is None else float(getattr(r, "od_pp_mm", 0.0)),
                    "" if getattr(r, "od_pp_rob_mm", None) is None else float(getattr(r, "od_pp_rob_mm", 0.0)),
                    "" if getattr(r, "id_pp_mm", None) is None else float(getattr(r, "id_pp_mm", 0.0)),
                    "" if getattr(r, "id_pp_rob_mm", None) is None else float(getattr(r, "id_pp_rob_mm", 0.0)),
                    str(getattr(r, "raw", "") or ""),
                ])

        raw_csv = run_dir / "raw_points.csv"
        pts = list(context.raw_points or [])
        with open(raw_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "serial", "run_id",
                "section_idx", "z_pos_mm", "sample_idx",
                "ts", "theta_deg", "bin",
                "phase",
                "od_mm", "id_mm", "cl_cnt",
                "raw_od", "raw_id",
            ])
            for p in pts:
                if not isinstance(p, dict):
                    continue
                w.writerow([
                    serial, run_id,
                    p.get("section_idx", ""),
                    p.get("z_pos_mm", ""),
                    p.get("sample_idx", ""),
                    p.get("ts", ""),
                    p.get("theta_deg", ""),
                    p.get("bin", ""),
                    p.get("phase", ""),
                    p.get("od_mm", ""),
                    (p.get("id_mm", "") if p.get("id_mm", None) not in (None, "") else p.get("id_out2_mm", "")),
                    p.get("cl_cnt", ""),
                    p.get("raw_od", ""),
                    p.get("raw_id", ""),
                ])

        meta_path = run_dir / "meta.json"
        meta = {
            "serial": serial,
            "run_id": run_id,
            "start_time": datetime.datetime.fromtimestamp(start_ts).isoformat(sep=" ", timespec="seconds"),
            "end_time": datetime.datetime.fromtimestamp(end_ts).isoformat(sep=" ", timespec="seconds"),
            "duration_s": float(end_ts - start_ts),
            "recipe": self._recipe_dump_dict(context.recipe),
            "device_code": self._get_device_code(),
            "software_version": self._software_version,
            "plc": {
                "ip": self._plc_info.get("ip", ""),
                "port": self._plc_info.get("port", ""),
                "unit": self._plc_info.get("unit", ""),
            },
            "gauge": {
                "enabled": self._gauge_info.get("enabled", True),
                "port": self._gauge_info.get("port", None),
            },
            "exports": {
                "section_results_csv": str(section_csv),
                "raw_points_csv": str(raw_csv),
                "meta_json": str(meta_path),
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        try:
            self.export_daily_summary(context)
        except Exception:
            pass

        return str(run_dir)

    def export_daily_summary(self, context: RunContext) -> None:
        serial = str(context.identity.serial or "")
        run_id = str(context.identity.run_id or "")
        if not serial or not run_id:
            return

        try:
            _start = float(context.identity.started_at_ts)
        except Exception:
            return
        try:
            _end = float(context.finished_at_ts if context.finished_at_ts is not None else time.time())
        except Exception:
            _end = float(time.time())

        try:
            day_dir = self._exports_root_dir() / datetime.date.fromtimestamp(_start).strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return

        summary_path = Path(day_dir) / "summary.csv"
        rcp = context.recipe
        try:
            recipe_name = str(getattr(rcp, "name", "") or "")
            od_std = getattr(rcp, "od_std_mm", None)
            id_std = getattr(rcp, "id_std_mm", None)
            od_tol = getattr(rcp, "od_tol_mm", None)
        except Exception:
            recipe_name = ""
            od_std, id_std, od_tol = None, None, None

        s = dict(context.summary or {})
        try:
            length_result = dict(context.length_result or {})
        except Exception:
            length_result = {}

        def _num(x: Any, fmt: str = "{:.3f}") -> str:
            try:
                if x is None:
                    return ""
                return fmt.format(float(x))
            except Exception:
                return ""

        try:
            id_mode = "single" if bool(getattr(rcp, "id_single_enable", False)) else "dual"
        except Exception:
            id_mode = "dual"
        if id_mode == "single":
            id_est_mm = _num(s.get("id_est_mm"))
            id_ecc_amp_mm = _num(s.get("id_ecc_amp_mm"))
            id_ecc_ang_deg = _num(s.get("id_ecc_ang_deg"), fmt="{:.2f}")
            id_pp_rob_mm = _num(s.get("id_pp_rob_mm"))
        else:
            id_est_mm = ""
            id_ecc_amp_mm = ""
            id_ecc_ang_deg = ""
            id_pp_rob_mm = ""

        header = [
            "date",
            "start_time",
            "end_time",
            "duration_s",
            "serial",
            "run_id",
            "recipe_name",
            "device_code",
            "od_std_mm",
            "id_std_mm",
            "od_tol_mm",
            "len_enabled",
            "len_skipped",
            "len_ok",
            "len_mm",
            "len_z_low",
            "len_z_high",
            "len_reason",
            "len_t_s",
            "straight_od_mm",
            "straight_id_mm",
            "axis_dist_mm",
            "conc_max_mm",
            "axis_span_max_mm",
            "od_tilt_deg",
            "od_end_off_mm",
            "od_slope_mm_per_mm",
            "id_tilt_deg",
            "id_end_off_mm",
            "id_slope_mm_per_mm",
            "max_od_dev_abs_mm",
            "max_id_dev_abs_mm",
            "max_od_round_mm",
            "max_id_round_mm",
            "max_od_pp_mm",
            "max_od_pp_rob_mm",
            "max_od_fit_res_mm",
            "od_range_mm",
            "id_range_mm",
            "od_mean_mm",
            "od_d_pp_mm",
            "od_e_mm",
            "id_mean_mm",
            "id_d_pp_mm",
            "id_mode",
            "id_est_mm",
            "id_ecc_amp_mm",
            "id_ecc_ang_deg",
            "id_pp_rob_mm",
            "split_shift_deg",
            "coax_unreliable",
            "summary_ok",
            "summary_reason",
            "status",
            "software_version",
        ]

        row = [
            datetime.date.fromtimestamp(_start).strftime("%Y-%m-%d"),
            datetime.datetime.fromtimestamp(_start).strftime("%H:%M:%S"),
            datetime.datetime.fromtimestamp(_end).strftime("%H:%M:%S"),
            f"{max(0.0, _end - _start):.3f}",
            serial,
            run_id,
            recipe_name,
            str(self._get_device_code() or ""),
            _num(od_std),
            _num(id_std),
            _num(od_tol),
            ("1" if bool(length_result.get("enabled", False)) else "0") if length_result else "",
            ("1" if bool(length_result.get("skipped", False)) else "0") if length_result else "",
            ("1" if bool(length_result.get("ok", False)) else "0") if length_result else "",
            _num(length_result.get("length_mm", None)) if length_result else "",
            _num(length_result.get("z_low", None)) if length_result else "",
            _num(length_result.get("z_high", None)) if length_result else "",
            str(length_result.get("reason", "") or "") if length_result else "",
            _num(length_result.get("t_s", None), fmt="{:.3f}") if length_result else "",
            _num(s.get("straight_od")),
            _num(s.get("straight_id")),
            _num(s.get("axis_dist")),
            _num(s.get("conc_max")),
            _num(s.get("axis_span_max")),
            _num(s.get("od_tilt_deg"), fmt="{:.4f}"),
            _num(s.get("od_end_off_mm")),
            _num(s.get("od_slope"), fmt="{:.6f}"),
            _num(s.get("id_tilt_deg"), fmt="{:.4f}"),
            _num(s.get("id_end_off_mm")),
            _num(s.get("id_slope"), fmt="{:.6f}"),
            _num(s.get("max_od_dev_abs")),
            _num(s.get("max_id_dev_abs")),
            _num(s.get("max_od_round")),
            _num(s.get("max_id_round")),
            _num(s.get("max_od_pp")),
            _num(s.get("max_od_pp_rob")),
            _num(s.get("max_od_fit_res")),
            _num(s.get("od_range")),
            _num(s.get("id_range")),
            _num(s.get("od_mean")),
            _num(s.get("od_d_pp")),
            _num(s.get("od_e")),
            _num(s.get("id_mean")),
            _num(s.get("id_d_pp")),
            id_mode,
            id_est_mm,
            id_ecc_amp_mm,
            id_ecc_ang_deg,
            id_pp_rob_mm,
            _num(s.get("split_shift_deg"), "{:.2f}"),
            ("1" if bool(s.get("coax_unreliable", False)) else "0") if (s.get("coax_unreliable") is not None) else "",
            "1" if bool(s.get("ok", False)) else "0",
            str(s.get("reason", "") or ""),
            str(context.status or "DONE"),
            self._software_version,
        ]

        try:
            existing_rows: list[list[str]] = []
            old_header: list[str] = []
            if summary_path.exists():
                with open(summary_path, "r", newline="", encoding="utf-8-sig") as f:
                    r = csv.reader(f)
                    existing_rows = [list(x) for x in r]
            if existing_rows:
                old_header = list(existing_rows[0])

            converted_rows: list[list[str]] = []
            if existing_rows and old_header and (old_header != header):
                old_map = {c: i for i, c in enumerate(old_header)}

                def _convert_one(rr: list[str]) -> list[str]:
                    out = ["" for _ in range(len(header))]
                    for j, col in enumerate(header):
                        i0 = old_map.get(col, None)
                        if i0 is None:
                            continue
                        try:
                            if i0 < len(rr):
                                out[j] = rr[i0]
                        except Exception:
                            pass
                    return out

                for rr in existing_rows[1:]:
                    try:
                        converted_rows.append(_convert_one(list(rr)))
                    except Exception:
                        converted_rows.append(["" for _ in range(len(header))])
            elif existing_rows and old_header and (old_header == header):
                converted_rows = [list(rr) for rr in existing_rows[1:]]

            out_rows: list[list[str]] = [header]
            try:
                run_id_col = header.index("run_id")
            except Exception:
                run_id_col = 5

            replaced = False
            for rr in converted_rows:
                try:
                    if len(rr) > run_id_col and str(rr[run_id_col]) == run_id:
                        out_rows.append(row)
                        replaced = True
                    else:
                        out_rows.append(rr)
                except Exception:
                    out_rows.append(rr)

            if not existing_rows:
                out_rows = [header, row]
            elif not replaced:
                out_rows.append(row)

            tmp = summary_path.with_suffix(".tmp")
            with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerows(out_rows)
            tmp.replace(summary_path)
        except Exception:
            try:
                new_file = not summary_path.exists()
                with open(summary_path, "a", newline="", encoding="utf-8-sig") as f:
                    w = csv.writer(f)
                    if new_file:
                        w.writerow(header)
                    w.writerow(row)
            except Exception:
                pass

    def _export_current_run(self, context: RunContext) -> tuple[bool, str]:
        try:
            run_dir = self.export_run(context)
            return True, f"已导出：{run_dir}"
        except Exception as e:
            return False, f"导出失败：{e}"

    def _export_daily_summary_csv(self, context: RunContext) -> None:
        self.export_daily_summary(context)


__all__ = ["RunRepository"]
