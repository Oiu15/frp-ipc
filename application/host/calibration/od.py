from __future__ import annotations

"""OD calibration mixin for AppHost."""

import datetime
import math
import tkinter as tk
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from config.addresses import CMD_EN_REQ


class HostOdCalibrationMixin:
    """Mixin providing OD calibration capture, defect handling, stats, and persistence."""

    calibration_repository: Any
    calibration_service: Any
    odcal_B_active_var: Any
    odcal_angle_src_var: Any
    odcal_cmd_var: Any
    odcal_defect_dyn_enable_var: Any
    odcal_defect_mode_var: Any
    odcal_defect_shift_var: Any
    odcal_defects_var: Any
    odcal_dref_var: Any
    odcal_drop_rate_var: Any
    odcal_filter_var: Any
    odcal_map_out1_var: Any
    odcal_msg_var: Any
    odcal_outlier_sigma_var: Any
    odcal_sum_max_var: Any
    odcal_sum_mean_var: Any
    odcal_sum_min_var: Any
    odcal_sum_std_var: Any
    _odcal_ax3_rotating: bool
    _odcal_defect_learn_A_data: Optional[dict]
    _odcal_defect_template_mask: list[int]
    _odcal_drop_cnt: int
    _odcal_points: list[dict]
    _odcal_rev_progress_deg: float
    _odcal_rev_target_deg: float
    _odcal_theta_last: Optional[float]
    _odcal_theta_start: Optional[float]
    _odcal_theta_unwrap: float

    if TYPE_CHECKING:
        def get_axis_copy(self, axis: int) -> Any: ...
        def set_cmd_bits(self, axis: int, set_mask: int = 0, clr_mask: int = 0) -> None: ...
        def _velmove_start_axis(self, axis: int, vel_velmove: float) -> None: ...
        def _velmove_stop_axis(self, axis: int) -> None: ...

    def learn_odcal_defect_a(self):
        return self._odcal_defect_learn_A()

    def learn_odcal_defect_b(self):
        return self._odcal_defect_learn_B()

    def clear_odcal_defect_template(self):
        return self._odcal_defect_clear_template()

    # =========================
    # OD Calibration (B)
    # =========================
    def _odcal_get_ax3_pos(self) -> Optional[float]:
        """Read AX3 act_pos (deg) from the latest PLC snapshot.

        Returns None if not available.
        """
        try:
            ac = self.get_axis_copy(3)
            return float(getattr(ac, "act_pos", 0.0) or 0.0)
        except Exception:
            return None

    def _odcal_update_rev_progress(self, theta_deg: Optional[float]) -> float:
        """Update internal unwrapped angle and return progress (deg) since start.

        Handles both continuous theta and 0..360 wrap-around.
        """
        if theta_deg is None:
            return float(self._odcal_rev_progress_deg or 0.0)

        try:
            th = float(theta_deg)
        except Exception:
            return float(self._odcal_rev_progress_deg or 0.0)

        if self._odcal_theta_start is None:
            self._odcal_theta_start = th
            self._odcal_theta_last = th
            self._odcal_theta_unwrap = th
            self._odcal_rev_progress_deg = 0.0
            return 0.0

        last = float(self._odcal_theta_last if self._odcal_theta_last is not None else th)
        dp = th - last
        # unwrap for 0..360 style angle
        if dp < -180.0:
            dp += 360.0
        elif dp > 180.0:
            dp -= 360.0

        self._odcal_theta_unwrap = float(self._odcal_theta_unwrap) + dp
        self._odcal_theta_last = th
        self._odcal_rev_progress_deg = float(self._odcal_theta_unwrap) - float(self._odcal_theta_start)
        return float(self._odcal_rev_progress_deg)

    def _odcal_rev_done(self) -> bool:
        """True if one-rev capture has reached target angle."""
        try:
            tgt = float(self._odcal_rev_target_deg or 360.0)
            prog = float(self._odcal_rev_progress_deg or 0.0)
            # tolerate small overshoot/undershoot
            return abs(prog) >= (tgt - 1.0)
        except Exception:
            return False

    def _odcal_start_ax3_rotation(self, speed_degps: float) -> None:
        """Start AX3 rotation using VelMove (deg/s)."""
        try:
            # Try to keep AX3 enabled during capture.
            try:
                self.set_cmd_bits(3, set_mask=CMD_EN_REQ, clr_mask=0)
            except Exception:
                pass
            self._velmove_start_axis(3, float(speed_degps))
            self._odcal_ax3_rotating = True
        except Exception:
            self._odcal_ax3_rotating = False
            raise

    def _odcal_stop_ax3_rotation(self) -> None:
        """Stop AX3 rotation if it was started by OD calibration."""
        try:
            if not bool(self._odcal_ax3_rotating):
                return
            self._velmove_stop_axis(3)
        finally:
            self._odcal_ax3_rotating = False




    def _odcal_deg_from_point(self, pt: dict) -> Optional[int]:
        """Return degree bin index [0..359] for a sample point.

        Priority:
        - theta_rel (one_rev 进度) -> 0..360+
        - theta (AX3 实际角度)   -> 任意，取 mod 360
        """
        try:
            th_rel = pt.get("theta_rel", None)
            if th_rel is not None:
                return int(math.floor(float(th_rel))) % 360
        except Exception:
            pass
        try:
            th = pt.get("theta", None)
            if th is not None:
                return int(math.floor(float(th))) % 360
        except Exception:
            pass
        return None

    def _odcal_bins_median(self, degs: list[int], vals: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """Median per 1° bin. Returns (bin_vals[360], has_data[360])."""
        bins = [[] for _ in range(360)]
        for d, v in zip(degs, vals):
            try:
                bins[int(d) % 360].append(float(v))
            except Exception:
                pass
        bin_vals = np.full((360,), np.nan, dtype=float)
        has = np.zeros((360,), dtype=bool)
        for i in range(360):
            if bins[i]:
                arr = np.array(bins[i], dtype=float)
                bin_vals[i] = float(np.median(arr))
                has[i] = True
        return bin_vals, has

    def _odcal_fit_harmonics(self, bin_vals: np.ndarray, has: np.ndarray, order: int = 3) -> np.ndarray:
        """Fit low-order harmonic model to bin_vals on bins with data.

        Model: y = a0 + Σ (ak cos(kθ) + bk sin(kθ)), k=1..order
        """
        try:
            idx = np.where(has)[0]
            if idx.size < max(8, 2 * order + 3):
                # too few points, fallback flat
                m = float(np.nanmedian(bin_vals)) if np.isfinite(np.nanmedian(bin_vals)) else 0.0
                return np.full((360,), m, dtype=float)
            th = np.deg2rad(idx.astype(float))
            cols = [np.ones_like(th)]
            for k in range(1, int(order) + 1):
                cols.append(np.cos(k * th))
                cols.append(np.sin(k * th))
            A = np.stack(cols, axis=1)  # (n, m)
            y = bin_vals[idx]
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            # predict all 360
            th_all = np.deg2rad(np.arange(360, dtype=float))
            cols_all = [np.ones_like(th_all)]
            for k in range(1, int(order) + 1):
                cols_all.append(np.cos(k * th_all))
                cols_all.append(np.sin(k * th_all))
            A_all = np.stack(cols_all, axis=1)
            yhat = A_all @ coef
            return yhat.astype(float)
        except Exception:
            m = float(np.nanmedian(bin_vals)) if np.isfinite(np.nanmedian(bin_vals)) else 0.0
            return np.full((360,), m, dtype=float)

    def _odcal_residual_bins(self, degs: list[int], sums: list[float], order: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (s_bin, r_bin, has)."""
        s_bin, has = self._odcal_bins_median(degs, sums)
        yhat = self._odcal_fit_harmonics(s_bin, has, order=order)
        r_bin = np.full((360,), np.nan, dtype=float)
        try:
            idx = np.where(has)[0]
            r_bin[idx] = s_bin[idx] - yhat[idx]
        except Exception:
            pass
        return s_bin, r_bin, has

    def _odcal_mask_to_ranges(self, mask: list[int]) -> list[tuple[int, int]]:
        """Convert 0/1 mask[360] to circular ranges (inclusive)."""
        if not mask or len(mask) != 360:
            return []
        m = [1 if int(x) else 0 for x in mask]
        if sum(m) <= 0:
            return []

        # find runs on [0..359]
        runs: list[tuple[int, int]] = []
        i = 0
        while i < 360:
            if m[i]:
                j = i
                while j + 1 < 360 and m[j + 1]:
                    j += 1
                runs.append((i, j))
                i = j + 1
            else:
                i += 1

        # merge wrap if needed (end + start)
        if len(runs) >= 2 and runs[0][0] == 0 and runs[-1][1] == 359:
            a0, b0 = runs[0]
            a1, b1 = runs[-1]
            runs = [(a1, b0)] + runs[1:-1]

        # normalize to [0..359], keep inclusive
        out = []
        for a, b in runs:
            out.append((int(a) % 360, int(b) % 360))
        return out

    def _odcal_ranges_to_mask(self, ranges: list[tuple[int, int]]) -> list[int]:
        m = [0] * 360
        for a, b in ranges or []:
            try:
                a = int(a) % 360
                b = int(b) % 360
            except Exception:
                continue
            if a <= b:
                for d in range(a, b + 1):
                    m[d] = 1
            else:
                for d in range(a, 360):
                    m[d] = 1
                for d in range(0, b + 1):
                    m[d] = 1
        return m

    def _odcal_ranges_str(self, ranges: list[tuple[int, int]]) -> str:
        if not ranges:
            return "--"
        parts = []
        for a, b in ranges:
            if a == b:
                parts.append(f"{a}°")
            else:
                parts.append(f"{a}~{b}°")
        return ", ".join(parts)

    def _odcal_shift_mask(self, mask: list[int], shift: int) -> list[int]:
        """Rotate mask into current-run coordinate: out[j] = mask[(j-shift) mod 360]."""
        if not mask or len(mask) != 360:
            return [0] * 360
        s = int(shift) % 360
        out = [0] * 360
        for j in range(360):
            out[j] = 1 if int(mask[(j - s) % 360]) else 0
        return out

    def _odcal_detect_defect_mask(
        self,
        r_bin: np.ndarray,
        has: np.ndarray,
        abs_thr: float = 0.010,
        k_sigma: float = 5.5,
        gap1_close: bool = True,
        min_len: int = 2,
        pad: int = 1,
        top_n: Optional[int] = None,
    ) -> tuple[list[int], dict]:
        """Detect negative dents in residual curve and return mask[360].

        Returns (mask, debug).
        """
        dbg: dict[str, Any] = {"abs_thr": float(abs_thr), "k_sigma": float(k_sigma)}
        try:
            idx = np.where(has & np.isfinite(r_bin))[0]
            if idx.size < 16:
                return [0] * 360, {"reason": "insufficient_bins", **dbg}

            rv = r_bin[idx].astype(float)
            med = float(np.median(rv))
            mad = float(np.median(np.abs(rv - med)))
            sigma = 1.4826 * mad
            thr = max(float(abs_thr), float(k_sigma) * float(sigma))
            dbg.update({"sigma_mad": float(sigma), "thr": float(thr)})

            cand = np.zeros((360,), dtype=bool)
            cand[idx] = (rv < (-thr))

            # close single-bin gaps: 1 0 1 -> 1 1 1
            if gap1_close:
                filled = cand.copy()
                for i in range(360):
                    if (not cand[i]) and cand[(i - 1) % 360] and cand[(i + 1) % 360]:
                        filled[i] = True
                cand = filled

            # find segments
            segs: list[tuple[int, int]] = []
            i = 0
            while i < 360:
                if cand[i]:
                    j = i
                    while j + 1 < 360 and cand[j + 1]:
                        j += 1
                    segs.append((i, j))
                    i = j + 1
                else:
                    i += 1
            # wrap merge
            if len(segs) >= 2 and segs[0][0] == 0 and segs[-1][1] == 359:
                a0, b0 = segs[0]
                a1, b1 = segs[-1]
                segs = [(a1, b0)] + segs[1:-1]

            # filter by min_len and score
            scored = []
            for a, b in segs:
                # length inclusive
                if a <= b:
                    degs = list(range(a, b + 1))
                else:
                    degs = list(range(a, 360)) + list(range(0, b + 1))
                if len(degs) < int(min_len):
                    continue
                # score: sum of negative depth beyond -thr
                sc = 0.0
                for d in degs:
                    if has[d] and np.isfinite(r_bin[d]):
                        sc += max(0.0, (-float(r_bin[d]) - thr))
                scored.append((sc, a, b, len(degs)))

            scored.sort(reverse=True, key=lambda x: x[0])
            if top_n is not None and top_n > 0:
                scored = scored[: int(top_n)]

            mask = np.zeros((360,), dtype=bool)
            kept_segs: list[tuple[int, int, float, int]] = []
            for sc, a, b, ln in scored:
                kept_segs.append((int(a), int(b), float(sc), int(ln)))
                # apply with padding
                if a <= b:
                    for d in range(a - pad, b + pad + 1):
                        mask[d % 360] = True
                else:
                    for d in list(range(a - pad, 360 + pad)) + list(range(0, b + pad + 1)):
                        mask[d % 360] = True

            dbg["segments"] = kept_segs
            return [1 if x else 0 for x in mask.tolist()], dbg
        except Exception as e:
            return [0] * 360, {"reason": f"exception:{e}", **dbg}

    def _odcal_best_shift_template(self, template_mask: list[int], r_bin: np.ndarray, has: np.ndarray) -> tuple[Optional[int], dict]:
        """Find circular shift that best aligns template dent positions to current residual."""
        dbg = {}
        if (not template_mask) or (len(template_mask) != 360) or (sum(int(x) for x in template_mask) <= 0):
            return None, {"reason": "no_template"}
        try:
            tpl_idx = [i for i, x in enumerate(template_mask) if int(x)]
            if len(tpl_idx) < 2:
                return None, {"reason": "template_too_small"}
            best_s = 0
            best_score = -1e30
            best_n = 0
            for s in range(360):
                sc = 0.0
                n = 0
                for i in tpl_idx:
                    j = (i + s) % 360
                    if bool(has[j]) and np.isfinite(r_bin[j]):
                        sc += (-float(r_bin[j]))  # prefer more negative
                        n += 1
                if n > 0:
                    sc = sc / n
                if sc > best_score:
                    best_score = sc
                    best_s = s
                    best_n = n
            dbg.update({"score": float(best_score), "n": int(best_n)})
            return int(best_s), dbg
        except Exception as e:
            return None, {"reason": f"exception:{e}"}

    def _odcal_best_shift_by_overlap(self, mask_a: list[int], mask_b: list[int]) -> tuple[Optional[int], dict]:
        """Find shift s maximizing overlap Σ a[i]*b[i+s]."""
        if (not mask_a) or (not mask_b) or (len(mask_a) != 360) or (len(mask_b) != 360):
            return None, {"reason": "bad_mask"}
        if sum(int(x) for x in mask_a) <= 0 or sum(int(x) for x in mask_b) <= 0:
            return None, {"reason": "empty_mask"}
        best_s = 0
        best_ov = -1
        for s in range(360):
            ov = 0
            for i in range(360):
                if int(mask_a[i]) and int(mask_b[(i + s) % 360]):
                    ov += 1
            if ov > best_ov:
                best_ov = ov
                best_s = s
        return int(best_s), {"overlap_bins": int(best_ov)}

    def _odcal_prepare_sums(self) -> tuple[list[float], dict]:
        """Prepare lL+lR list for B computation.

        Pipeline:
        1) raw sum series from points (optionally mapped OUT1/OUT2)
        2) optional dent masking (TEMPLATE aligned, or DYNAMIC if enabled and no template)
        3) optional median filter on time series
        4) optional outlier removal by sigma
        """
        meta: dict = {
            "defect_mode": "OFF",
            "defect_shift": None,
            "defect_masked": 0,
            "defect_debug": {},
        }

        pts = list(getattr(self, "_odcal_points", []) or [])
        if not pts:
            return [], meta

        # map OUT1/OUT2 to L/R
        out1_map = (self.odcal_map_out1_var.get() or "L").strip().upper()
        map_swap = (out1_map == "R")

        sums_raw: list[float] = []
        degs_raw: list[int] = []
        deg_missing = 0

        for pt in pts:
            try:
                v1 = float(pt.get("v1", 0.0))
                v2 = float(pt.get("v2", 0.0))
            except Exception:
                continue

            # swap mapping if OUT1->R
            if map_swap:
                v1, v2 = v2, v1

            sums_raw.append(v1 + v2)

            d = self._odcal_deg_from_point(pt)
            if d is None:
                deg_missing += 1
                degs_raw.append(0)
            else:
                degs_raw.append(int(d) % 360)

        # ------------------------------
        # 2) Dent masking
        # ------------------------------
        have_angle = (deg_missing == 0) and (len(degs_raw) >= 16)
        template_loaded = bool(getattr(self, "_odcal_defect_template_mask", None)) and (sum(int(x) for x in self._odcal_defect_template_mask) > 0)

        # default UI: show template ranges when idle
        try:
            if template_loaded and (self.odcal_defects_var.get() in ("--", "", None)):
                self.odcal_defects_var.set("模板: " + self._odcal_ranges_str(self._odcal_mask_to_ranges(self._odcal_defect_template_mask)))
        except Exception:
            pass

        masked_idx = set()

        if have_angle and template_loaded:
            try:
                _, r_bin, has = self._odcal_residual_bins(degs_raw, sums_raw, order=3)
                shift, sdbg = self._odcal_best_shift_template(self._odcal_defect_template_mask, r_bin, has)
                meta["defect_debug"] = {"align": sdbg}
                if shift is not None:
                    run_mask = self._odcal_shift_mask(self._odcal_defect_template_mask, shift)
                    for i, d in enumerate(degs_raw):
                        if int(run_mask[int(d) % 360]):
                            masked_idx.add(i)
                    meta.update({"defect_mode": "TEMPLATE", "defect_shift": int(shift), "defect_masked": int(len(masked_idx))})
                    # UI hints
                    try:
                        self.odcal_defect_mode_var.set("TEMPLATE")
                        self.odcal_defect_shift_var.set(f"{int(shift)}°")
                        self.odcal_defects_var.set("屏蔽: " + self._odcal_ranges_str(self._odcal_mask_to_ranges(run_mask)))
                    except Exception:
                        pass
            except Exception as e:
                meta["defect_debug"] = {"reason": f"template_exception:{e}"}

        # dynamic fallback (only when no template)
        if have_angle and (not template_loaded):
            try:
                dyn_en = int(getattr(self, "odcal_defect_dyn_enable_var", tk.IntVar(value=0)).get() or 0)
            except Exception:
                dyn_en = 0
            if dyn_en:
                try:
                    _, r_bin, has = self._odcal_residual_bins(degs_raw, sums_raw, order=3)
                    dyn_mask, ddbg = self._odcal_detect_defect_mask(r_bin, has, top_n=1)
                    meta["defect_debug"] = {"dynamic": ddbg}
                    if sum(int(x) for x in dyn_mask) > 0:
                        for i, d in enumerate(degs_raw):
                            if int(dyn_mask[int(d) % 360]):
                                masked_idx.add(i)
                        meta.update({"defect_mode": "DYNAMIC", "defect_shift": None, "defect_masked": int(len(masked_idx))})
                        try:
                            self.odcal_defect_mode_var.set("DYNAMIC")
                            self.odcal_defect_shift_var.set("--")
                            self.odcal_defects_var.set("屏蔽: " + self._odcal_ranges_str(self._odcal_mask_to_ranges(dyn_mask)))
                        except Exception:
                            pass
                except Exception as e:
                    meta["defect_debug"] = {"reason": f"dynamic_exception:{e}"}

        # apply masking to time series
        if masked_idx:
            sums1 = [s for i, s in enumerate(sums_raw) if i not in masked_idx]
        else:
            sums1 = list(sums_raw)

        # ------------------------------
        # 3) optional median filter on time series
        # ------------------------------
        try:
            mode = str(self.odcal_filter_var.get() if hasattr(self, "odcal_filter_var") else "无")
        except Exception:
            mode = "无"

        if mode.startswith("中值"):
            try:
                win = 3 if "3" in mode else 5
                if win >= 3 and len(sums1) >= win:
                    arr = np.array(sums1, dtype=float)
                    out = []
                    half = win // 2
                    for i in range(len(arr)):
                        a = max(0, i - half)
                        b = min(len(arr), i + half + 1)
                        out.append(float(np.median(arr[a:b])))
                    sums1 = out
            except Exception:
                pass

        # ------------------------------
        # 4) outlier removal (sigma)
        # ------------------------------
        try:
            sig = float(self.odcal_outlier_sigma_var.get() if hasattr(self, "odcal_outlier_sigma_var") else 0.0)
        except Exception:
            sig = 0.0

        sums2 = sums1
        if sig and sig > 0 and len(sums1) >= 8:
            try:
                arr = np.array(sums1, dtype=float)
                m = float(np.mean(arr))
                sd = float(np.std(arr))
                if sd > 1e-12:
                    keep = np.abs(arr - m) <= (sig * sd)
                    sums2 = [float(v) for v, k in zip(arr.tolist(), keep.tolist()) if k]
            except Exception:
                sums2 = sums1

        # finalize OFF mode UI if no masking
        if not masked_idx:
            try:
                if template_loaded:
                    self.odcal_defect_mode_var.set("TEMPLATE")
                    self.odcal_defects_var.set("模板: " + self._odcal_ranges_str(self._odcal_mask_to_ranges(self._odcal_defect_template_mask)))
                else:
                    self.odcal_defect_mode_var.set("OFF")
                    self.odcal_defect_shift_var.set("--")
                    # keep last shown, but if empty show --
                    if (self.odcal_defects_var.get() or "").strip() == "":
                        self.odcal_defects_var.set("--")
            except Exception:
                pass

        return sums2, meta





    def _odcal_defect_learn_A(self):
        """Record run-A residual/mask as a learning baseline."""
        try:
            if not getattr(self, "_odcal_points", None):
                self.odcal_msg_var.set("学习A：无采样数据")
                return
            # require angle
            degs = []
            sums = []
            miss = 0
            out1_map = (self.odcal_map_out1_var.get() or "L").strip().upper()
            swap = (out1_map == "R")
            for pt in self._odcal_points:
                d = self._odcal_deg_from_point(pt)
                if d is None:
                    miss += 1
                    continue
                try:
                    v1 = float(pt.get("v1", 0.0))
                    v2 = float(pt.get("v2", 0.0))
                except Exception:
                    continue
                if swap:
                    v1, v2 = v2, v1
                degs.append(int(d) % 360)
                sums.append(v1 + v2)

            if miss > 0 or len(degs) < 64:
                self.odcal_msg_var.set("学习A：需要一圈角度数据（建议 one_rev + 角度=AX3）")
                return

            _, r_bin, has = self._odcal_residual_bins(degs, sums, order=3)
            mask, dbg = self._odcal_detect_defect_mask(r_bin, has, top_n=None)
            if sum(int(x) for x in mask) <= 0:
                self.odcal_msg_var.set("学习A：未检测到明显凹陷（阈值过严或数据不足）")
                return

            self._odcal_defect_learn_A_data = {
                "r_bin": r_bin.tolist(),
                "has": has.tolist(),
                "mask": mask,
                "dbg": dbg,
                "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }
            self.odcal_defect_mode_var.set("LEARN_A")
            self.odcal_defect_shift_var.set("--")
            self.odcal_defects_var.set("A: " + self._odcal_ranges_str(self._odcal_mask_to_ranges(mask)))
            self.odcal_msg_var.set("学习A：已记录。请转动/翻转环规后再采集，点击“学习B(生成表)”")
        except Exception as e:
            self.odcal_msg_var.set(f"学习A失败: {e}")

    def _odcal_defect_learn_B(self):
        """Use current run as B, align to A, and generate a stable template mask."""
        try:
            A = getattr(self, "_odcal_defect_learn_A_data", None)
            if not A:
                self.odcal_msg_var.set("学习B：请先完成学习A")
                return
            if not getattr(self, "_odcal_points", None):
                self.odcal_msg_var.set("学习B：无采样数据")
                return

            # current (B)
            degs = []
            sums = []
            miss = 0
            out1_map = (self.odcal_map_out1_var.get() or "L").strip().upper()
            swap = (out1_map == "R")
            for pt in self._odcal_points:
                d = self._odcal_deg_from_point(pt)
                if d is None:
                    miss += 1
                    continue
                try:
                    v1 = float(pt.get("v1", 0.0))
                    v2 = float(pt.get("v2", 0.0))
                except Exception:
                    continue
                if swap:
                    v1, v2 = v2, v1
                degs.append(int(d) % 360)
                sums.append(v1 + v2)

            if miss > 0 or len(degs) < 64:
                self.odcal_msg_var.set("学习B：需要一圈角度数据（建议 one_rev + 角度=AX3）")
                return

            _, rB, hasB = self._odcal_residual_bins(degs, sums, order=3)
            maskB, dbgB = self._odcal_detect_defect_mask(rB, hasB, top_n=None)
            if sum(int(x) for x in maskB) <= 0:
                self.odcal_msg_var.set("学习B：未检测到明显凹陷（阈值过严或数据不足）")
                return

            maskA = list(A.get("mask") or [0] * 360)
            if len(maskA) != 360:
                self.odcal_msg_var.set("学习B：A 数据异常")
                return

            # align B to A by overlap
            shift_ab, sdbg = self._odcal_best_shift_by_overlap(maskA, maskB)
            if shift_ab is None:
                self.odcal_msg_var.set("学习B：对齐失败（A/B 凹陷段过少）")
                return

            # build mean residual in A-frame and re-detect on mean (more robust than intersection)
            rA = np.array(A.get("r_bin") or [np.nan] * 360, dtype=float)
            hasA = np.array(A.get("has") or [False] * 360, dtype=bool)

            rB_shift = np.full((360,), np.nan, dtype=float)
            hasB_shift = np.zeros((360,), dtype=bool)
            for i in range(360):
                j = (i + int(shift_ab)) % 360
                rB_shift[i] = rB[j] if np.isfinite(rB[j]) else np.nan
                hasB_shift[i] = bool(hasB[j])

            # mean
            rM = np.full((360,), np.nan, dtype=float)
            hasM = hasA | hasB_shift
            for i in range(360):
                if not hasM[i]:
                    continue
                vals = []
                if bool(hasA[i]) and np.isfinite(rA[i]):
                    vals.append(float(rA[i]))
                if bool(hasB_shift[i]) and np.isfinite(rB_shift[i]):
                    vals.append(float(rB_shift[i]))
                if vals:
                    rM[i] = float(sum(vals) / len(vals))

            maskT, dbgT = self._odcal_detect_defect_mask(rM, hasM, top_n=None)
            if sum(int(x) for x in maskT) <= 0:
                # fallback: intersection after alignment
                maskB_inA = [int(maskB[(i + int(shift_ab)) % 360]) for i in range(360)]
                maskT = [1 if (int(maskA[i]) and int(maskB_inA[i])) else 0 for i in range(360)]
                dbgT = {"fallback": "intersection"}

            # persist template
            self._odcal_defect_template_mask = [1 if int(x) else 0 for x in maskT]
            ranges = self._odcal_mask_to_ranges(self._odcal_defect_template_mask)

            # update calibration json (preserve existing fields)
            data = self.calibration_repository.load_od_active()
            data["defects"] = {
                "template_mask": list(self._odcal_defect_template_mask),
                "template_ranges": [[a, b] for a, b in ranges],
                "learn_shift_ab_deg": int(shift_ab),
                "learned_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "learn_dbg": {"A": A.get("dbg", {}), "B": dbgB, "align": sdbg, "template": dbgT},
            }
            self._odcal_save_active(data)

            # clear A buffer
            self._odcal_defect_learn_A_data = None

            self.odcal_defect_mode_var.set("TEMPLATE")
            self.odcal_defect_shift_var.set("--")
            self.odcal_defects_var.set("模板: " + self._odcal_ranges_str(ranges))
            self.odcal_msg_var.set(f"凹陷表已生成：{self._odcal_ranges_str(ranges)}（A<-B shift={int(shift_ab)}°）")
        except Exception as e:
            self.odcal_msg_var.set(f"学习B失败: {e}")

    def _odcal_defect_clear_template(self):
        """Remove persisted defect template."""
        try:
            self._odcal_defect_template_mask = [0] * 360
            self._odcal_defect_learn_A_data = None

            data = self.calibration_repository.load_od_active()
            if "defects" in data:
                data.pop("defects", None)
            self._odcal_save_active(data)

            self.odcal_defect_mode_var.set("OFF")
            self.odcal_defect_shift_var.set("--")
            self.odcal_defects_var.set("--")
            self.odcal_msg_var.set("已清除凹陷表")
        except Exception as e:
            self.odcal_msg_var.set(f"清除失败: {e}")




    def _odcal_on_gauge_sample(self, payload: dict):
        return self.calibration_service.on_od_gauge_sample(self, payload)

    def _odcal_update_stats(self):
        try:
            n = len(self._odcal_points)
            if n <= 0:
                return
            sums, meta = self._odcal_prepare_sums()
            if not sums:
                # only update counts
                dr = (self._odcal_drop_cnt / max(1, n))
                self.odcal_drop_rate_var.set(f"{dr*100:.1f}%")
                return

            mean_sum = sum(sums) / len(sums)
            # std
            var = sum((x - mean_sum) ** 2 for x in sums) / max(1, (len(sums) - 1))
            std_sum = math.sqrt(var)
            self.odcal_sum_mean_var.set(f"{mean_sum:.5f}")
            self.odcal_sum_std_var.set(f"{std_sum:.5f}")
            self.odcal_sum_min_var.set(f"{min(sums):.5f}")
            self.odcal_sum_max_var.set(f"{max(sums):.5f}")

            dr = (self._odcal_drop_cnt / max(1, n))
            self.odcal_drop_rate_var.set(f"{dr*100:.1f}%")
        except Exception:
            pass

    # =========================

    # ------------------------------
    # OD Calibration (B) persistence
    # ------------------------------



    def _odcal_save_active(self, data: dict) -> None:
        self.calibration_repository.save_od_active(data or {})

    def _odcal_load_active(self) -> None:
        data = self.calibration_repository.load_od_prefill()
        try:
            b = data.get("B_active", None)
            if b is not None:
                self.odcal_B_active_var.set(f"{float(b):.5f}")
        except Exception:
            pass

        # Also prefill UI inputs for convenience
        try:
            dref = data.get("D_ref", None)
            if dref is not None:
                self.odcal_dref_var.set(f"{float(dref):.3f}")
        except Exception:
            pass
        try:
            cmd = str(data.get("cmd_used", "") or "").strip()
            if cmd:
                self.odcal_cmd_var.set(cmd)
        except Exception:
            pass
        try:
            out1 = str(data.get("out1_map", "L") or "L").upper()
            if out1 in ("L", "R"):
                self.odcal_map_out1_var.set(out1)
        except Exception:
            pass

        # Prefill advanced params if present
        try:
            ang = str(data.get("angle_src_ui", "") or "").strip()
            if ang:
                self.odcal_angle_src_var.set(ang)
            flt = str(data.get("filter", "") or "").strip()
            if flt:
                self.odcal_filter_var.set(flt)
            sig = data.get("outlier_sigma", None)
            if sig is not None:
                self.odcal_outlier_sigma_var.set(str(sig))
        except Exception:
            pass
        # load defect template (if any)
        try:
            self._odcal_defect_template_mask = list(data.get("defect_template_mask", [0] * 360) or [0] * 360)

            if sum(int(x) for x in (self._odcal_defect_template_mask or [])) > 0:
                self.odcal_defect_mode_var.set("TEMPLATE")
                self.odcal_defect_shift_var.set("--")
                self.odcal_defects_var.set("模板: " + self._odcal_ranges_str(self._odcal_mask_to_ranges(self._odcal_defect_template_mask)))
            else:
                self.odcal_defect_mode_var.set("OFF")
                self.odcal_defect_shift_var.set("--")
                self.odcal_defects_var.set("--")
        except Exception:
            pass



