from __future__ import annotations

"""Validation export repository.

This repository keeps validation-mode exports separate from formal production
measurement exports while preserving the same day/serial directory style.
"""

import csv
import datetime
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from application.contracts import ValidationRepositoryProtocol
from application.state import ValidationExportContext
from core.models import Recipe


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
            'scan_mode': str(getattr(recipe, 'scan_mode', 'sync') or 'sync'),
            'od_std_mm': float(getattr(recipe, 'od_std_mm', 0.0) or 0.0),
            'id_std_mm': float(getattr(recipe, 'id_std_mm', 0.0) or 0.0),
        }

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
