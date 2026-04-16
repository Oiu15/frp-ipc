from __future__ import annotations

"""Validation export repository.

This repository keeps validation-mode exports separate from formal production
measurement exports while preserving the same day/serial directory style.
"""

import csv
import datetime
import json
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from application.contracts import ValidationRepositoryProtocol
from application.state import ValidationExportContext
from core.models import MeasureRow, Recipe
from frp_workflow.validation_workflow import (
    FixedSectionRepeatCapture,
    FixedSectionRepeatabilityRequest,
    FixedSectionRepeatRow,
    FixedSectionWindow,
)

_FIXED_SECTION_EXPORT_SCHEMA_VERSION = 'validation_fixed_section_v1'


class ValidationRepository(ValidationRepositoryProtocol):
    """Filesystem-backed repository for validation-mode exports."""

    def __init__(self, *, app_root_dir: Path | None = None, software_version: str = "") -> None:
        self._app_root_dir_override = Path(app_root_dir) if app_root_dir is not None else None
        self._software_version = str(software_version or "")

    def _app_root_dir(self) -> Path:
        try:
            if self._app_root_dir_override is not None:
                return self._app_root_dir_override
            return Path.home() / 'FRP_IPC'
        except Exception:
            return Path('./FRP_IPC')

    def _exports_root_dir(self) -> Path:
        return self._app_root_dir() / 'validation_exports'

    def _recipe_dump_dict(self, recipe: Recipe) -> dict[str, Any]:
        return {
            'name': str(getattr(recipe, 'name', '') or ''),
            'section_count': int(getattr(recipe, 'section_count', 0) or 0),
            'section_sampling_mode': str(getattr(recipe, 'section_sampling_mode', getattr(recipe, 'scan_mode', 'sync')) or 'sync'),
            'sampling_window_mode': str(getattr(recipe, 'sampling_window_mode', 'shared') or 'shared'),
            'scan_mode': str(getattr(recipe, 'scan_mode', 'sync') or 'sync'),
            'od_std_mm': float(getattr(recipe, 'od_std_mm', 0.0) or 0.0),
            'id_std_mm': float(getattr(recipe, 'id_std_mm', 0.0) or 0.0),
        }

    @staticmethod
    def _path_map(paths: Mapping[str, Path]) -> dict[str, str]:
        return {str(name): str(path) for name, path in paths.items()}

    @staticmethod
    def _path_names(paths: Mapping[str, Path]) -> list[str]:
        return [path.name for _, path in paths.items()]

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dict(payload), f, ensure_ascii=False, indent=2)

    @staticmethod
    def _fixed_section_request_dict(request: FixedSectionRepeatabilityRequest) -> dict[str, Any]:
        return dict(asdict(request))

    def export_run(self, context: ValidationExportContext) -> str:
        start_ts = float(context.started_at_ts if context.started_at_ts is not None else context.identity.started_at_ts)
        end_ts = float(context.finished_at_ts if context.finished_at_ts is not None else time.time())
        serial = str(context.identity.serial)
        run_id = str(context.identity.run_id)

        day_dir = self._exports_root_dir() / datetime.date.fromtimestamp(start_ts).strftime('%Y-%m-%d')
        day_dir.mkdir(parents=True, exist_ok=True)
        run_dir = day_dir / serial
        run_dir.mkdir(parents=True, exist_ok=True)

        result_path = run_dir / 'validation_result.json'
        events_path = run_dir / 'validation_events.json'
        payload = {
            'serial': serial,
            'run_id': run_id,
            'start_time': datetime.datetime.fromtimestamp(start_ts).isoformat(sep=' ', timespec='seconds'),
            'end_time': datetime.datetime.fromtimestamp(end_ts).isoformat(sep=' ', timespec='seconds'),
            'duration_s': float(end_ts - start_ts),
            'status': str(context.status or ''),
            'message': str(context.message or ''),
            'standard_piece_id': context.standard_piece_id,
            'validation_batch_id': context.validation_batch_id,
            'repeat_measurement_count': int(context.repeat_measurement_count or 0),
            'recipe': self._recipe_dump_dict(context.recipe),
            'calibration': asdict(context.calibration),
            'summary': dict(context.summary or {}),
            'exports': {
                'validation_result_json': str(result_path),
                'validation_events_json': str(events_path),
            },
            'software_version': self._software_version,
        }
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        with open(events_path, 'w', encoding='utf-8') as f:
            json.dump(list(context.events or []), f, ensure_ascii=False, indent=2)

        try:
            self.export_daily_summary(context)
        except Exception:
            pass

        return str(run_dir)

    def export_fixed_section_repeatability(
        self,
        *,
        context: ValidationExportContext,
        request: FixedSectionRepeatabilityRequest,
        rows: list[FixedSectionRepeatRow],
        summary: Mapping[str, Any],
        captures: Sequence[FixedSectionRepeatCapture] | None = None,
    ) -> str:
        start_ts = float(context.started_at_ts if context.started_at_ts is not None else context.identity.started_at_ts)
        end_ts = float(context.finished_at_ts if context.finished_at_ts is not None else time.time())
        serial = str(context.identity.serial)
        first_row = rows[0] if rows else None

        day_dir = self._exports_root_dir() / datetime.date.fromtimestamp(start_ts).strftime('%Y-%m-%d')
        day_dir.mkdir(parents=True, exist_ok=True)
        run_dir = day_dir / serial
        run_dir.mkdir(parents=True, exist_ok=True)

        meta_path = run_dir / 'validation_meta.json'
        result_path = run_dir / 'validation_result.json'
        repeat_results_path = run_dir / 'repeat_results.csv'
        rows_path = run_dir / 'repeat_rows.csv'
        summary_path = run_dir / 'repeat_summary.json'
        section_results_path = run_dir / 'repeat_section_results.csv'
        raw_points_path = run_dir / 'repeat_raw_points.csv'
        windows_path = run_dir / 'repeat_windows.csv'
        fit_results_path = run_dir / 'repeat_fit_results.csv'
        events_path = run_dir / 'validation_events.json'

        canonical_paths = {
            'validation_result_json': result_path,
            'validation_meta_json': meta_path,
            'validation_events_json': events_path,
            'repeat_results_csv': repeat_results_path,
            'repeat_raw_points_csv': raw_points_path,
            'repeat_fit_results_csv': fit_results_path,
        }
        legacy_paths = {
            'repeat_rows_csv': rows_path,
            'repeat_section_results_csv': section_results_path,
            'repeat_windows_csv': windows_path,
            'repeat_summary_json': summary_path,
        }
        combined_paths = {
            **canonical_paths,
            **legacy_paths,
        }

        summary_payload = dict(summary or {})
        summary_payload.setdefault('task_name', str(request.task_name or 'fixed_section_repeatability'))
        summary_payload.setdefault(
            'section_name',
            str(first_row.section_name if first_row is not None else request.section_name or ''),
        )
        summary_payload.setdefault(
            'measure_section_index',
            None if first_row is None else first_row.measure_section_index,
        )
        summary_payload.setdefault(
            'measure_section_name',
            str(first_row.measure_section_name if first_row is not None else ''),
        )
        summary_payload.setdefault(
            'measured_z_pos_mm',
            None if first_row is None else float(first_row.measured_z_pos_mm),
        )
        summary_payload.setdefault('metric_name', str(request.metric_name or ''))
        summary_payload.setdefault('repeat_count', len(rows))
        if captures:
            summary_payload.setdefault('section_result_count', int(len(captures)))
            summary_payload.setdefault('raw_point_count', int(sum(len(capture.raw_points) for capture in captures)))
            summary_payload.setdefault('window_count', int(sum(len(capture.windows) for capture in captures)))

        meta_payload: dict[str, Any] = {
            'schema_version': _FIXED_SECTION_EXPORT_SCHEMA_VERSION,
            'validation_kind': str(request.task_name or 'fixed_section_repeatability'),
            'flow_kind': str(request.task_name or 'fixed_section_repeatability'),
            'task_name': str(request.task_name or 'fixed_section_repeatability'),
            'serial': serial,
            'validation_batch_id': context.validation_batch_id,
            'started_at_ts': start_ts,
            'finished_at_ts': end_ts,
            'start_time': datetime.datetime.fromtimestamp(start_ts).isoformat(sep=' ', timespec='seconds'),
            'end_time': datetime.datetime.fromtimestamp(end_ts).isoformat(sep=' ', timespec='seconds'),
            'duration_s': float(max(0.0, end_ts - start_ts)),
            'status': str(context.status or ''),
            'message': str(context.message or ''),
            'requested_section_name': str(request.section_name or ''),
            'section_name': str(first_row.section_name if first_row is not None else request.section_name or ''),
            'measure_section_index': (None if first_row is None else first_row.measure_section_index),
            'measure_section_name': str(first_row.measure_section_name if first_row is not None else ''),
            'measured_z_pos_mm': (None if first_row is None else float(first_row.measured_z_pos_mm)),
            'metric_name': str(request.metric_name or ''),
            'reclamp_between_repeats': bool(getattr(request, 'reclamp_between_repeats', False)),
            'move_enabled': bool(getattr(request, 'move_enabled', False)),
            'move_channel': str(getattr(request, 'move_channel', '') or ''),
            'move_scenario': str(getattr(request, 'move_scenario', '') or ''),
            'move_away_delta_mm': float(getattr(request, 'move_away_delta_mm', 0.0) or 0.0),
            'move_from_section_index': int(getattr(request, 'move_from_section_index', 1) or 1),
            'move_target_section_index': int(getattr(request, 'move_target_section_index', 1) or 1),
            'move_return_section_index': int(getattr(request, 'move_return_section_index', 1) or 1),
            'position_settle_s': float(getattr(request, 'position_settle_s', 0.0) or 0.0),
            'sample_delay_s': float(getattr(request, 'sample_delay_s', 0.0) or 0.0),
            'repeat_count': len(rows),
            'requested_repeat_count': int(getattr(request, 'repeat_count', 0) or 0),
            'completed_repeat_count': len(rows),
            'recipe': self._recipe_dump_dict(context.recipe),
            'calibration': asdict(context.calibration),
            'request': self._fixed_section_request_dict(request),
            'summary': dict(summary_payload),
            'canonical_outputs': self._path_names(canonical_paths),
            'legacy_outputs': self._path_names(legacy_paths),
            'deprecated_outputs': self._path_names(legacy_paths),
            'canonical_exports': self._path_map(canonical_paths),
            'legacy_exports': self._path_map(legacy_paths),
            'exports': self._path_map(combined_paths),
            'software_version': self._software_version,
        }
        if getattr(context.identity, 'run_id', None):
            meta_payload['run_id'] = str(context.identity.run_id)

        self._write_json(meta_path, meta_payload)
        with open(events_path, 'w', encoding='utf-8') as f:
            json.dump(list(context.events or []), f, ensure_ascii=False, indent=2)
        self._export_repeat_results(repeat_results_path, rows)
        self._export_repeat_results(rows_path, rows)
        self._write_json(summary_path, summary_payload)

        captures_list = list(captures or [])
        self._export_repeat_section_results(section_results_path, captures_list)
        self._export_repeat_raw_points(raw_points_path, captures_list)
        self._export_repeat_windows(windows_path, captures_list)
        self._export_repeat_fit_results(fit_results_path, captures_list)

        result_payload: dict[str, Any] = {
            'schema_version': _FIXED_SECTION_EXPORT_SCHEMA_VERSION,
            'validation_kind': str(request.task_name or 'fixed_section_repeatability'),
            'flow_kind': str(request.task_name or 'fixed_section_repeatability'),
            'serial': serial,
            'run_id': str(getattr(context.identity, 'run_id', '') or ''),
            'started_at_ts': start_ts,
            'finished_at_ts': end_ts,
            'start_time': datetime.datetime.fromtimestamp(start_ts).isoformat(sep=' ', timespec='seconds'),
            'end_time': datetime.datetime.fromtimestamp(end_ts).isoformat(sep=' ', timespec='seconds'),
            'duration_s': float(max(0.0, end_ts - start_ts)),
            'status': str(context.status or ''),
            'message': str(context.message or ''),
            'standard_piece_id': context.standard_piece_id,
            'validation_batch_id': context.validation_batch_id,
            'repeat_measurement_count': len(rows),
            'requested_repeat_count': int(getattr(request, 'repeat_count', 0) or 0),
            'config': {
                'request': self._fixed_section_request_dict(request),
                'recipe': self._recipe_dump_dict(context.recipe),
            },
            'final_summary': dict(summary_payload),
            'summary': dict(summary_payload),
            'canonical_outputs': self._path_names(canonical_paths),
            'legacy_outputs': self._path_names(legacy_paths),
            'canonical_exports': self._path_map(canonical_paths),
            'legacy_exports': self._path_map(legacy_paths),
            'exports': self._path_map(combined_paths),
            'software_version': self._software_version,
        }
        self._write_json(result_path, result_payload)

        return str(run_dir)

    def _export_repeat_results(
        self,
        path: Path,
        rows: Sequence[FixedSectionRepeatRow],
    ) -> None:
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                'repeat_index',
                'section_name',
                'measure_section_index',
                'measure_section_name',
                'measured_z_pos_mm',
                'metric_name',
                'measured_value_mm',
                'settle_s_used',
                'sample_delay_s_used',
                'capture_start_ts',
                'capture_end_ts',
                'measured_at_ts',
            ])
            for row in rows:
                writer.writerow([
                    int(row.repeat_index),
                    str(row.section_name),
                    ('' if row.measure_section_index is None else int(row.measure_section_index)),
                    str(row.measure_section_name),
                    f'{float(row.measured_z_pos_mm):.3f}',
                    str(row.metric_name),
                    f'{float(row.measured_value_mm):.3f}',
                    f'{float(row.settle_s_used):.3f}',
                    f'{float(row.sample_delay_s_used):.3f}',
                    ('' if row.capture_start_ts is None else f'{float(row.capture_start_ts):.6f}'),
                    ('' if row.capture_end_ts is None else f'{float(row.capture_end_ts):.6f}'),
                    f'{float(row.measured_at_ts):.3f}',
                ])

    def _export_repeat_section_results(
        self,
        path: Path,
        captures: Sequence[FixedSectionRepeatCapture],
    ) -> None:
        row_field_names = [field.name for field in fields(MeasureRow)]
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                'repeat_index',
                'section_name',
                'measure_section_index',
                'measure_section_name',
                'measured_z_pos_mm',
                'metric_name',
                'measured_value_mm',
                'settle_s_used',
                'sample_delay_s_used',
                'capture_start_ts',
                'capture_end_ts',
                'measured_at_ts',
                *row_field_names,
            ])
            for capture in captures:
                row_dict = asdict(capture.section_result)
                writer.writerow([
                    int(capture.repeat_index),
                    str(capture.section_name),
                    ('' if capture.measure_section_index is None else int(capture.measure_section_index)),
                    str(capture.measure_section_name),
                    f'{float(capture.measured_z_pos_mm):.3f}',
                    str(capture.metric_name),
                    f'{float(capture.measured_value_mm):.6f}',
                    f'{float(capture.settle_s_used):.3f}',
                    f'{float(capture.sample_delay_s_used):.3f}',
                    ('' if capture.capture_start_ts is None else f'{float(capture.capture_start_ts):.6f}'),
                    ('' if capture.capture_end_ts is None else f'{float(capture.capture_end_ts):.6f}'),
                    f'{float(capture.measured_at_ts):.6f}',
                    *[row_dict.get(field_name) for field_name in row_field_names],
                ])

    def _export_repeat_windows(
        self,
        path: Path,
        captures: Sequence[FixedSectionRepeatCapture],
    ) -> None:
        window_field_names = [field.name for field in fields(FixedSectionWindow)]
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(window_field_names)
            for capture in captures:
                for window in capture.windows:
                    window_dict = asdict(window)
                    writer.writerow([window_dict.get(field_name) for field_name in window_field_names])

    @staticmethod
    def _format_optional_float(value: Any, *, precision: int) -> str:
        if value is None:
            return ''
        try:
            return f'{float(value):.{int(precision)}f}'
        except Exception:
            return ''

    @staticmethod
    def _format_optional_int(value: Any) -> str:
        if value is None:
            return ''
        try:
            return str(int(value))
        except Exception:
            return ''

    def _export_repeat_fit_results(
        self,
        path: Path,
        captures: Sequence[FixedSectionRepeatCapture],
    ) -> None:
        field_names = [
            'repeat_index',
            'measure_section_index',
            'measure_section_name',
            'measured_z_pos_mm',
            'od_center_x_mm',
            'od_center_y_mm',
            'od_radius_mm',
            'od_diameter_fit_mm',
            'id_center_x_mm',
            'id_center_y_mm',
            'id_radius_mm',
            'id_diameter_fit_mm',
            'od_ecc_mm',
            'id_ecc_mm',
            'concentricity_mm',
        ]
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(field_names)
            for capture in captures:
                fit_result = getattr(capture, 'fit_result', None)
                writer.writerow([
                    int(capture.repeat_index),
                    self._format_optional_int(
                        capture.measure_section_index
                        if fit_result is None
                        else fit_result.measure_section_index
                    ),
                    str(
                        capture.measure_section_name
                        if fit_result is None
                        else fit_result.measure_section_name
                    ),
                    self._format_optional_float(
                        capture.measured_z_pos_mm
                        if fit_result is None
                        else fit_result.measured_z_pos_mm,
                        precision=3,
                    ),
                    self._format_optional_float(
                        None if fit_result is None else fit_result.od_center_x_mm,
                        precision=6,
                    ),
                    self._format_optional_float(
                        None if fit_result is None else fit_result.od_center_y_mm,
                        precision=6,
                    ),
                    self._format_optional_float(
                        None if fit_result is None else fit_result.od_radius_mm,
                        precision=6,
                    ),
                    self._format_optional_float(
                        None if fit_result is None else fit_result.od_diameter_fit_mm,
                        precision=6,
                    ),
                    self._format_optional_float(
                        None if fit_result is None else fit_result.id_center_x_mm,
                        precision=6,
                    ),
                    self._format_optional_float(
                        None if fit_result is None else fit_result.id_center_y_mm,
                        precision=6,
                    ),
                    self._format_optional_float(
                        None if fit_result is None else fit_result.id_radius_mm,
                        precision=6,
                    ),
                    self._format_optional_float(
                        None if fit_result is None else fit_result.id_diameter_fit_mm,
                        precision=6,
                    ),
                    self._format_optional_float(
                        None if fit_result is None else fit_result.od_ecc_mm,
                        precision=6,
                    ),
                    self._format_optional_float(
                        None if fit_result is None else fit_result.id_ecc_mm,
                        precision=6,
                    ),
                    self._format_optional_float(
                        None if fit_result is None else fit_result.concentricity_mm,
                        precision=6,
                    ),
                ])

    def _export_repeat_raw_points(
        self,
        path: Path,
        captures: Sequence[FixedSectionRepeatCapture],
    ) -> None:
        base_fields = [
            'repeat_index',
            'section_name',
            'measure_section_index',
            'measure_section_name',
            'measured_z_pos_mm',
            'metric_name',
            'measured_at_ts',
            'window_index',
            'window_role',
            'point_index_in_window',
            'sample_idx',
            'section_idx',
            'z_pos_mm',
            'phase',
            'ts',
            'theta_deg',
            'bin',
            'od_mm',
            'id_mm',
            'id_out2_mm',
            'id_c_mm',
            'id_m_mm',
            'od_delta',
            'raw_od',
            'raw_id',
        ]
        extra_keys: set[str] = set()
        for capture in captures:
            for point in capture.raw_points:
                if isinstance(point, Mapping):
                    extra_keys.update(str(key) for key in point.keys())
        field_names = base_fields + sorted(key for key in extra_keys if key not in base_fields)

        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(field_names)
            for capture in captures:
                for point in capture.raw_points:
                    point_dict = dict(point)
                    point_dict.setdefault('repeat_index', int(capture.repeat_index))
                    point_dict.setdefault('section_name', str(capture.section_name))
                    point_dict.setdefault('measure_section_index', capture.measure_section_index)
                    point_dict.setdefault('measure_section_name', str(capture.measure_section_name))
                    point_dict.setdefault('measured_z_pos_mm', float(capture.measured_z_pos_mm))
                    point_dict.setdefault('metric_name', str(capture.metric_name))
                    point_dict.setdefault('measured_at_ts', float(capture.measured_at_ts))
                    writer.writerow([point_dict.get(field_name) for field_name in field_names])

    def export_daily_summary(self, context: ValidationExportContext) -> None:
        serial = str(context.identity.serial or '')
        run_id = str(context.identity.run_id or '')
        if not serial or not run_id:
            return

        try:
            start_ts = float(context.started_at_ts if context.started_at_ts is not None else context.identity.started_at_ts)
        except Exception:
            return
        try:
            end_ts = float(context.finished_at_ts if context.finished_at_ts is not None else time.time())
        except Exception:
            end_ts = float(time.time())

        day_dir = self._exports_root_dir() / datetime.date.fromtimestamp(start_ts).strftime('%Y-%m-%d')
        day_dir.mkdir(parents=True, exist_ok=True)
        summary_path = day_dir / 'summary.csv'

        header = [
            'date',
            'start_time',
            'end_time',
            'duration_s',
            'serial',
            'run_id',
            'recipe_name',
            'standard_piece_id',
            'validation_batch_id',
            'repeat_measurement_count',
            'status',
            'message',
            'summary_json',
        ]
        row = [
            datetime.date.fromtimestamp(start_ts).strftime('%Y-%m-%d'),
            datetime.datetime.fromtimestamp(start_ts).strftime('%H:%M:%S'),
            datetime.datetime.fromtimestamp(end_ts).strftime('%H:%M:%S'),
            f'{max(0.0, end_ts - start_ts):.3f}',
            serial,
            run_id,
            str(getattr(context.recipe, 'name', '') or ''),
            str(context.standard_piece_id or ''),
            str(context.validation_batch_id or ''),
            str(int(context.repeat_measurement_count or 0)),
            str(context.status or ''),
            str(context.message or ''),
            json.dumps(dict(context.summary or {}), ensure_ascii=False, sort_keys=True),
        ]

        existing_rows: list[list[str]] = []
        if summary_path.exists():
            try:
                with open(summary_path, 'r', newline='', encoding='utf-8-sig') as f:
                    existing_rows = [list(r) for r in csv.reader(f)]
            except Exception:
                existing_rows = []

        out_rows = [header]
        run_id_col = header.index('run_id')
        replaced = False
        for rr in existing_rows[1:]:
            if len(rr) > run_id_col and str(rr[run_id_col]) == run_id:
                out_rows.append(row)
                replaced = True
            else:
                out_rows.append(rr)
        if not existing_rows:
            out_rows = [header, row]
        elif not replaced:
            out_rows.append(row)

        tmp = summary_path.with_suffix('.tmp')
        with open(tmp, 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerows(out_rows)
        tmp.replace(summary_path)


__all__ = ['ValidationRepository']
