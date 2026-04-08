from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import math

from core.models import MeasureRow, Recipe


@dataclass(frozen=True, slots=True)
class SummarySnapshot:
    straight_od: float | None = None
    straight_id: float | None = None
    axis_dist: float | None = None
    conc_max: float | None = None
    axis_span_max: float | None = None
    od_tilt_deg: float | None = None
    od_end_off_mm: float | None = None
    od_slope: float | None = None
    id_tilt_deg: float | None = None
    id_end_off_mm: float | None = None
    id_slope: float | None = None
    provided_fields: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class EccentricityUpdate:
    row_index: int
    od_ecc: float | None
    id_ecc: float | None


class ResultsService:
    """Result-domain logic extracted from UI event handling."""

    def summary_snapshot_from_payload(self, payload: Any) -> SummarySnapshot:
        p = payload if isinstance(payload, Mapping) else {}
        provided_fields: set[str] = set()

        def _provided(*keys: str) -> bool:
            return any(key in p for key in keys)

        def _pull(*keys: str) -> float | None:
            for key in keys:
                if key in p:
                    return self._to_float(p.get(key))
            return None

        if _provided('straight_od', 'straightness'):
            provided_fields.add('straight_od')
        if _provided('straight_id'):
            provided_fields.add('straight_id')
        if _provided('axis_dist'):
            provided_fields.add('axis_dist')
        if _provided('conc_max'):
            provided_fields.add('conc_max')
        if _provided('axis_span_max'):
            provided_fields.add('axis_span_max')
        if _provided('od_tilt_deg'):
            provided_fields.add('od_tilt_deg')
        if _provided('od_end_off_mm'):
            provided_fields.add('od_end_off_mm')
        if _provided('od_slope'):
            provided_fields.add('od_slope')
        if _provided('id_tilt_deg'):
            provided_fields.add('id_tilt_deg')
        if _provided('id_end_off_mm'):
            provided_fields.add('id_end_off_mm')
        if _provided('id_slope'):
            provided_fields.add('id_slope')

        return SummarySnapshot(
            straight_od=_pull('straight_od', 'straightness'),
            straight_id=_pull('straight_id'),
            axis_dist=_pull('axis_dist'),
            conc_max=_pull('conc_max'),
            axis_span_max=_pull('axis_span_max'),
            od_tilt_deg=_pull('od_tilt_deg'),
            od_end_off_mm=_pull('od_end_off_mm'),
            od_slope=_pull('od_slope'),
            id_tilt_deg=_pull('id_tilt_deg'),
            id_end_off_mm=_pull('id_end_off_mm'),
            id_slope=_pull('id_slope'),
            provided_fields=frozenset(provided_fields),
        )

    def merge_summary_snapshot(
        self,
        summary_cache: Mapping[str, Any] | None,
        snapshot: SummarySnapshot,
    ) -> dict[str, Any]:
        merged = dict(summary_cache or {})
        for field_name in snapshot.provided_fields:
            merged[field_name] = getattr(snapshot, field_name)
        return merged

    def build_eccentricity_updates(self, payload: Any) -> list[EccentricityUpdate]:
        p = payload if isinstance(payload, Mapping) else {}
        ecc_od = list(p.get('ecc_od', []) or [])
        ecc_id = list(p.get('ecc_id', []) or [])
        updates: list[EccentricityUpdate] = []
        for idx, (od_raw, id_raw) in enumerate(zip(ecc_od, ecc_id)):
            updates.append(
                EccentricityUpdate(
                    row_index=int(idx),
                    od_ecc=self._to_float(od_raw),
                    id_ecc=self._to_float(id_raw),
                )
            )
        return updates

    def apply_eccentricity_updates(
        self,
        rows: Sequence[MeasureRow],
        updates: Sequence[EccentricityUpdate],
    ) -> None:
        for update in updates:
            try:
                row = rows[int(update.row_index)]
            except Exception:
                continue
            try:
                row.od_ecc = update.od_ecc
            except Exception:
                pass
            try:
                row.id_ecc = update.id_ecc
            except Exception:
                pass

    def compute_run_summary(
        self,
        *,
        recipe: Recipe,
        rows: Sequence[MeasureRow],
        raw_points: Sequence[Mapping[str, Any]],
        summary_cache: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        rows_list = list(rows or [])
        if not rows_list:
            return {'ok': False, 'reason': 'No section results'}

        seed = dict(summary_cache or {})

        od_dev_abs_vals: list[float] = []
        id_dev_abs_vals: list[float] = []
        od_dev_vals: list[float] = []
        id_dev_vals: list[float] = []
        od_round_vals: list[float] = []
        id_round_vals: list[float] = []
        od_avg_vals: list[float] = []
        od_runout_vals: list[float] = []
        id_avg_vals: list[float] = []
        conc_vals: list[float] = []
        split_shift_abs_vals: list[float] = []
        coax_unreliable_any = False
        od_pp_vals: list[float] = []
        od_pp_rob_vals: list[float] = []
        od_fit_res_vals: list[float] = []
        id_pp_rob_vals: list[float] = []
        id_ecc_amp_vals: list[float] = []
        id_ecc_ang_vals: list[tuple[float, float]] = []
        judge_total = 0
        judge_ok_cnt = 0

        for row in rows_list:
            try:
                judge_total += 1
                if bool(getattr(row, 'ok', True)):
                    judge_ok_cnt += 1
            except Exception:
                pass

            od_dev = self._to_float(getattr(row, 'od_dev', None))
            id_dev = self._to_float(getattr(row, 'id_dev', None))
            od_round = self._to_float(getattr(row, 'od_round_fit_rob_mm', None))
            if od_round is None:
                od_round = self._to_float(getattr(row, 'od_round', None))
            id_round = self._to_float(getattr(row, 'id_round_fit_rob_mm', None))
            if id_round is None:
                id_round = self._to_float(getattr(row, 'id_round', None))
            id_avg = self._to_float(getattr(row, 'id_avg', None))
            od_avg = self._to_float(getattr(row, 'od_avg', None))
            od_runout = self._to_float(getattr(row, 'od_runout', None))
            conc = self._to_float(getattr(row, 'concentricity', None))

            od_pp = self._to_float(getattr(row, 'od_pp_mm', None))
            if od_pp is None:
                od_pp = self._to_float(getattr(row, 'od_round', None))
            od_pp_rob = self._to_float(getattr(row, 'od_pp_rob_mm', None))
            od_fit_res = self._to_float(getattr(row, 'od_round_fit_rob_mm', None))
            if od_fit_res is None:
                od_fit_res = self._to_float(getattr(row, 'od_round_fit_mm', None))
            id_pp_rob = self._to_float(getattr(row, 'id_pp_rob_mm', None))
            if id_pp_rob is None:
                id_pp_rob = self._to_float(getattr(row, 'id_round', None))
            id_ecc_amp = self._to_float(getattr(row, 'id_e', None))
            id_ecc_ang = self._to_float(getattr(row, 'id_phi_deg', None))

            if od_dev is not None:
                od_dev_abs_vals.append(abs(od_dev))
                od_dev_vals.append(od_dev)
            if id_dev is not None:
                id_dev_abs_vals.append(abs(id_dev))
                id_dev_vals.append(id_dev)
            if od_round is not None:
                od_round_vals.append(od_round)
            if id_round is not None:
                id_round_vals.append(id_round)
            if od_avg is not None:
                od_avg_vals.append(od_avg)
            if id_avg is not None:
                id_avg_vals.append(id_avg)
            if od_runout is not None:
                od_runout_vals.append(od_runout)
            if conc is not None:
                conc_vals.append(conc)

            split_shift = self._to_float(getattr(row, 'split_shift_deg', None))
            if split_shift is not None:
                split_shift_abs_vals.append(abs(split_shift))
            try:
                if bool(getattr(row, 'coax_unreliable', False)):
                    coax_unreliable_any = True
            except Exception:
                pass

            if od_pp is not None:
                od_pp_vals.append(od_pp)
            if od_pp_rob is not None:
                od_pp_rob_vals.append(od_pp_rob)
            if od_fit_res is not None:
                od_fit_res_vals.append(od_fit_res)
            if id_pp_rob is not None:
                id_pp_rob_vals.append(id_pp_rob)
            if id_ecc_amp is not None:
                id_ecc_amp_vals.append(id_ecc_amp)
                if id_ecc_ang is not None:
                    id_ecc_ang_vals.append((id_ecc_amp, id_ecc_ang))

        if not (od_dev_abs_vals or id_dev_abs_vals or od_round_vals or id_round_vals or od_avg_vals or od_runout_vals or conc_vals):
            return {'ok': False, 'reason': 'No valid data'}

        max_od_dev = max(od_dev_abs_vals) if od_dev_abs_vals else None
        max_id_dev = max(id_dev_abs_vals) if id_dev_abs_vals else None
        max_od_round = max(od_round_vals) if od_round_vals else None
        max_id_round = max(id_round_vals) if id_round_vals else None
        max_od_pp = max(od_pp_vals) if od_pp_vals else None
        max_od_pp_rob = max(od_pp_rob_vals) if od_pp_rob_vals else None
        max_od_fit_res = max(od_fit_res_vals) if od_fit_res_vals else None
        max_id_pp_rob = max(id_pp_rob_vals) if id_pp_rob_vals else None
        max_id_ecc_amp = max(id_ecc_amp_vals) if id_ecc_amp_vals else None
        id_ecc_ang_deg = None
        try:
            if id_ecc_ang_vals:
                id_ecc_ang_deg = float(max(id_ecc_ang_vals, key=lambda item: float(item[0]))[1])
        except Exception:
            id_ecc_ang_deg = None

        od_mean = (sum(od_avg_vals) / len(od_avg_vals)) if od_avg_vals else None
        od_d_pp = float(max_od_round) if max_od_round is not None else None
        id_mean = (sum(id_avg_vals) / len(id_avg_vals)) if id_avg_vals else None
        id_d_pp = float(max_id_round) if max_id_round is not None else None
        conc_max = max(conc_vals) if conc_vals else None

        od_range = None
        id_range = None
        use_id_raw = not bool(getattr(recipe, 'id_single_enable', False))
        od_vals: list[float] = []
        id_vals: list[float] = []
        for point in raw_points or []:
            if not isinstance(point, Mapping):
                continue
            od_v = self._to_float(point.get('od_mm'))
            if od_v is not None and od_v > 0:
                od_vals.append(od_v)
            if use_id_raw:
                id_v = self._to_float(point.get('id_mm'))
                if id_v is not None and id_v > 0:
                    id_vals.append(id_v)
        if len(od_vals) >= 2:
            od_range = float(max(od_vals) - min(od_vals))
        if len(id_vals) >= 2 and use_id_raw:
            id_range = float(max(id_vals) - min(id_vals))
        if od_range is None and od_dev_vals:
            od_range = float(max(od_dev_vals) - min(od_dev_vals))
        if id_range is None and id_dev_vals:
            id_range = float(max(id_dev_vals) - min(id_dev_vals))

        od_e = None
        try:
            if bool(getattr(recipe, 'od_use_edges', False)) and od_runout_vals:
                od_e = float(max(od_runout_vals)) / 2.0
        except Exception:
            od_e = None

        return {
            'ok': True,
            'reason': '',
            'max_od_dev_abs': float(max_od_dev) if max_od_dev is not None else None,
            'max_id_dev_abs': float(max_id_dev) if max_id_dev is not None else None,
            'max_od_round': float(max_od_round) if max_od_round is not None else None,
            'max_id_round': float(max_id_round) if max_id_round is not None else None,
            'split_shift_deg': float(max(split_shift_abs_vals)) if split_shift_abs_vals else None,
            'coax_unreliable': bool(coax_unreliable_any) if (split_shift_abs_vals or coax_unreliable_any) else None,
            'max_od_pp': float(max_od_pp) if max_od_pp is not None else None,
            'max_od_pp_rob': float(max_od_pp_rob) if max_od_pp_rob is not None else None,
            'max_od_fit_res': float(max_od_fit_res) if max_od_fit_res is not None else None,
            'od_mean': float(od_mean) if od_mean is not None else None,
            'od_d_pp': float(od_d_pp) if od_d_pp is not None else None,
            'od_e': float(od_e) if od_e is not None else None,
            'od_range': float(od_range) if od_range is not None else None,
            'id_mean': float(id_mean) if id_mean is not None else None,
            'id_d_pp': float(id_d_pp) if id_d_pp is not None else None,
            'id_est_mm': float(id_mean) if id_mean is not None else None,
            'id_pp_rob_mm': float(max_id_pp_rob) if max_id_pp_rob is not None else None,
            'id_ecc_amp_mm': float(max_id_ecc_amp) if max_id_ecc_amp is not None else None,
            'id_ecc_ang_deg': float(id_ecc_ang_deg) if id_ecc_ang_deg is not None else None,
            'id_range': float(id_range) if id_range is not None else None,
            'straight_od': self._to_float(seed.get('straight_od')),
            'straight_id': self._to_float(seed.get('straight_id')),
            'axis_dist': self._to_float(seed.get('axis_dist')),
            'conc_max': float(conc_max) if conc_max is not None else self._to_float(seed.get('conc_max')),
            'axis_span_max': self._to_float(seed.get('axis_span_max')),
            'od_tilt_deg': self._to_float(seed.get('od_tilt_deg')),
            'od_end_off_mm': self._to_float(seed.get('od_end_off_mm')),
            'od_slope': self._to_float(seed.get('od_slope')),
            'id_tilt_deg': self._to_float(seed.get('id_tilt_deg')),
            'id_end_off_mm': self._to_float(seed.get('id_end_off_mm')),
            'id_slope': self._to_float(seed.get('id_slope')),
            'judge_ok_cnt': int(judge_ok_cnt),
            'judge_total': int(judge_total),
        }

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            result = float(value)
            if not math.isfinite(result):
                return None
            return result
        except Exception:
            return None


__all__ = ['EccentricityUpdate', 'ResultsService', 'SummarySnapshot']
