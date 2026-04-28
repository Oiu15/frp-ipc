from __future__ import annotations

"""Manual export support for completed production measurement history."""

import datetime as _dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SECTION_COUNT = 5
HISTORY_INDEX_FILENAME = "history_index.json"
HISTORY_INDEX_SCHEMA_VERSION = 1

DETECTION_SUMMARY_COLUMNS = (
    "编号",
    "流水号",
    "平均外径",
    "截面1外径真圆度",
    "截面2外径真圆度",
    "截面3外径真圆度",
    "截面4外径真圆度",
    "截面5外径真圆度",
    "外圆直线度",
    "外径极差",
    "平均内径",
    "截面1内径真圆度",
    "截面2内径真圆度",
    "截面3内径真圆度",
    "截面4内径真圆度",
    "截面5内径真圆度",
    "内圆直线度",
    "截面1同心度",
    "截面2同心度",
    "截面3同心度",
    "截面4同心度",
    "截面5同心度",
)

_REQUIRED_SECTION_COLUMNS = {
    "section_idx",
    "od_pp_rob_mm",
    "id_round_fit_mm",
    "concentricity_mm",
}

_REQUIRED_SUMMARY_COLUMNS = {
    "od_end_off_mm",
    "id_end_off_mm",
    "od_range_mm",
}


@dataclass(frozen=True, slots=True)
class HistoryExportEntry:
    date: str
    serial: str
    run_id: str
    start_time: str
    recipe_name: str
    status: str
    run_dir: Path
    section_results_csv: Path
    summary_csv: Path
    meta_json: Path
    sort_ts: float


class HistoryResultExportService:
    """Scan completed production exports and build detection summary workbooks."""

    def __init__(self, *, app_root_dir: Path | None = None) -> None:
        self._app_root_dir_override = Path(app_root_dir) if app_root_dir is not None else None

    def _app_root_dir(self) -> Path:
        if self._app_root_dir_override is not None:
            return self._app_root_dir_override
        try:
            return Path.home() / "FRP_IPC"
        except Exception:
            return Path("./FRP_IPC")

    def _exports_root_dir(self) -> Path:
        return self._app_root_dir() / "exports"

    def _history_index_path(self) -> Path:
        return self._exports_root_dir() / HISTORY_INDEX_FILENAME

    def list_exportable_entries(self) -> list[HistoryExportEntry]:
        indexed = self._load_history_index_entries()
        if indexed is not None:
            return indexed
        return self.rebuild_history_index()

    def rebuild_history_index(self) -> list[HistoryExportEntry]:
        entries = self._scan_exportable_entries_from_files()
        if self._exports_root_dir().exists():
            self._write_history_index_items([self._entry_to_index_item(entry) for entry in entries])
        return entries

    def upsert_history_index_entry(
        self,
        *,
        date: str,
        serial: str,
        run_id: str,
        start_time: str,
        recipe_name: str,
        status: str,
        run_dir: Path,
        section_results_csv: Path,
        summary_csv: Path,
        meta_json: Path,
        completed: bool,
        completed_sections: int,
        expected_sections: int,
        section_count: int | None = None,
    ) -> None:
        section_count = completed_sections if section_count is None else section_count
        item = {
            "date": str(date or ""),
            "serial": str(serial or ""),
            "run_id": str(run_id or ""),
            "start_time": str(start_time or ""),
            "recipe_name": str(recipe_name or ""),
            "status": str(status or ""),
            "completed": bool(completed),
            "completed_sections": int(completed_sections or 0),
            "expected_sections": int(expected_sections or 0),
            "section_count": int(section_count or 0),
            "run_dir": self._relative_path_text(run_dir),
            "section_results_csv": self._relative_path_text(section_results_csv),
            "summary_csv": self._relative_path_text(summary_csv),
            "meta_json": self._relative_path_text(meta_json),
            "sort_ts": self._parse_time_key(str(date or ""), str(start_time or "")),
        }
        item["exportable"] = self._index_item_is_exportable(item)
        self._upsert_history_index_item(item)

    def _scan_exportable_entries_from_files(self) -> list[HistoryExportEntry]:
        entries: list[HistoryExportEntry] = []
        exports_root = self._exports_root_dir()
        if not exports_root.exists():
            return entries

        for day_dir in sorted((p for p in exports_root.iterdir() if p.is_dir()), key=lambda p: p.name):
            summary_csv = day_dir / "summary.csv"
            if not summary_csv.exists():
                continue
            try:
                summary_df = self._read_csv(summary_csv)
            except Exception:
                continue
            for run_dir in sorted((p for p in day_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
                entry = self._entry_from_run_dir(day_dir.name, run_dir, summary_csv, summary_df)
                if entry is not None:
                    entries.append(entry)

        entries.sort(key=lambda item: (item.date, item.sort_ts, item.serial))
        return entries

    def _load_history_index_entries(self) -> list[HistoryExportEntry] | None:
        index_path = self._history_index_path()
        if not index_path.exists():
            return None
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        if int(data.get("schema_version", 0) or 0) != HISTORY_INDEX_SCHEMA_VERSION:
            return None
        raw_entries = data.get("entries", [])
        if not isinstance(raw_entries, list):
            return None

        entries: list[HistoryExportEntry] = []
        for raw in raw_entries:
            if not isinstance(raw, dict) or not self._truthy(raw.get("exportable")):
                continue
            entry = self._entry_from_index_item(raw)
            if entry is not None:
                entries.append(entry)
        entries.sort(key=lambda item: (item.date, item.sort_ts, item.serial))
        return entries

    def _entry_from_index_item(self, item: dict[str, Any]) -> HistoryExportEntry | None:
        date_text = str(item.get("date") or "").strip()
        serial = str(item.get("serial") or "").strip()
        if not date_text or not serial:
            return None
        run_id = str(item.get("run_id") or "").strip()
        start_text = str(item.get("start_time") or "").strip()
        status = str(item.get("status") or "DONE").strip().upper() or "DONE"
        run_dir = self._path_from_index_text(item.get("run_dir") or f"exports/{date_text}/{serial}")
        summary_csv = self._path_from_index_text(item.get("summary_csv") or f"exports/{date_text}/summary.csv")
        section_results_csv = self._path_from_index_text(
            item.get("section_results_csv") or f"exports/{date_text}/{serial}/section_results.csv"
        )
        meta_json = self._path_from_index_text(item.get("meta_json") or f"exports/{date_text}/{serial}/meta.json")
        try:
            sort_ts = float(item.get("sort_ts", self._parse_time_key(date_text, start_text)) or 0.0)
        except Exception:
            sort_ts = self._parse_time_key(date_text, start_text)
        return HistoryExportEntry(
            date=date_text,
            serial=serial,
            run_id=run_id,
            start_time=start_text or date_text,
            recipe_name=str(item.get("recipe_name") or "").strip(),
            status=status,
            run_dir=run_dir,
            section_results_csv=section_results_csv,
            summary_csv=summary_csv,
            meta_json=meta_json,
            sort_ts=sort_ts,
        )

    def _entry_to_index_item(self, entry: HistoryExportEntry) -> dict[str, Any]:
        return {
            "date": entry.date,
            "serial": entry.serial,
            "run_id": entry.run_id,
            "start_time": entry.start_time,
            "recipe_name": entry.recipe_name,
            "status": entry.status,
            "completed": True,
            "completed_sections": SECTION_COUNT,
            "expected_sections": SECTION_COUNT,
            "section_count": SECTION_COUNT,
            "exportable": True,
            "run_dir": self._relative_path_text(entry.run_dir),
            "section_results_csv": self._relative_path_text(entry.section_results_csv),
            "summary_csv": self._relative_path_text(entry.summary_csv),
            "meta_json": self._relative_path_text(entry.meta_json),
            "sort_ts": float(entry.sort_ts or 0.0),
        }

    def _upsert_history_index_item(self, item: dict[str, Any]) -> None:
        index_path = self._history_index_path()
        try:
            data = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
        except Exception:
            data = {}
        if not isinstance(data, dict) or int(data.get("schema_version", 0) or 0) != HISTORY_INDEX_SCHEMA_VERSION:
            entries: list[dict[str, Any]] = []
        else:
            raw_entries = data.get("entries", [])
            entries = [dict(x) for x in raw_entries if isinstance(x, dict)] if isinstance(raw_entries, list) else []

        item_key = self._index_match_key(item)
        updated: list[dict[str, Any]] = []
        replaced = False
        for old in entries:
            if self._index_match_key(old) == item_key:
                updated.append(dict(item))
                replaced = True
            else:
                updated.append(old)
        if not replaced:
            updated.append(dict(item))
        self._write_history_index_items(updated)

    def _index_match_key(self, item: dict[str, Any]) -> tuple[str, str, str]:
        run_id = str(item.get("run_id") or "").strip()
        if run_id:
            return ("run_id", run_id, "")
        return ("serial", str(item.get("date") or "").strip(), str(item.get("serial") or "").strip())

    def _write_history_index_items(self, items: list[dict[str, Any]]) -> None:
        index_path = self._history_index_path()
        data = {
            "schema_version": HISTORY_INDEX_SCHEMA_VERSION,
            "updated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "entries": items,
        }
        try:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = index_path.with_name(f"{index_path.name}.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(index_path)
        except Exception:
            pass

    def _index_item_is_exportable(self, item: dict[str, Any]) -> bool:
        if not self._truthy(item.get("completed")):
            return False
        try:
            completed_sections = int(item.get("completed_sections", 0) or 0)
            section_count = int(item.get("section_count", completed_sections) or 0)
        except Exception:
            return False
        if completed_sections < SECTION_COUNT or section_count < SECTION_COUNT:
            return False
        for key in ("meta_json", "section_results_csv", "summary_csv"):
            try:
                if not self._path_from_index_text(item.get(key, "")).exists():
                    return False
            except Exception:
                return False
        return True

    def _relative_path_text(self, path: Path) -> str:
        p = Path(path)
        try:
            return str(p.resolve().relative_to(self._app_root_dir().resolve())).replace("\\", "/")
        except Exception:
            return str(p)

    def _path_from_index_text(self, value: Any) -> Path:
        text = str(value or "").strip()
        if not text:
            return self._app_root_dir()
        p = Path(text)
        if p.is_absolute():
            return p
        return self._app_root_dir() / p

    def export_detection_summary(self, entries: list[HistoryExportEntry], output_path: Path) -> Path:
        if not entries:
            raise ValueError("没有选择可导出的测量记录。")

        pd = self._pandas()
        rows = [self._build_output_row(index, entry) for index, entry in enumerate(entries, start=1)]
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows, columns=list(DETECTION_SUMMARY_COLUMNS))
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="检测数据汇总", index=False)
        return output_path

    def _entry_from_run_dir(
        self,
        date_text: str,
        run_dir: Path,
        summary_csv: Path,
        summary_df: Any,
    ) -> HistoryExportEntry | None:
        meta_json = run_dir / "meta.json"
        section_results_csv = run_dir / "section_results.csv"
        if not meta_json.exists() or not section_results_csv.exists():
            return None

        try:
            meta = json.loads(meta_json.read_text(encoding="utf-8")) or {}
            if not isinstance(meta, dict):
                return None
        except Exception:
            return None

        serial = str(meta.get("serial") or run_dir.name or "").strip()
        run_id = str(meta.get("run_id") or "").strip()
        if not serial:
            return None

        summary_row = self._find_summary_row(summary_df, serial=serial, run_id=run_id)
        if summary_row is None:
            return None
        if any(self._number_or_blank(self._series_value(summary_row, col)) == "" for col in _REQUIRED_SUMMARY_COLUMNS):
            return None

        status = str(meta.get("status") or self._series_value(summary_row, "status") or "").strip().upper()
        if not self._is_complete_meta(meta, status):
            return None

        try:
            if not self._section_results_are_complete(section_results_csv, serial=serial, run_id=run_id):
                return None
        except Exception:
            return None

        start_text = str(meta.get("start_time") or "").strip()
        if not start_text:
            start_text = self._display_start_time(date_text, summary_row)
        recipe_name = str(self._series_value(summary_row, "recipe_name") or "").strip()
        if not recipe_name:
            try:
                recipe = meta.get("recipe", {})
                if isinstance(recipe, dict):
                    recipe_name = str(recipe.get("name") or "").strip()
            except Exception:
                recipe_name = ""

        return HistoryExportEntry(
            date=str(date_text),
            serial=serial,
            run_id=run_id,
            start_time=start_text,
            recipe_name=recipe_name,
            status=status or "DONE",
            run_dir=run_dir,
            section_results_csv=section_results_csv,
            summary_csv=summary_csv,
            meta_json=meta_json,
            sort_ts=self._parse_time_key(date_text, start_text),
        )

    def _build_output_row(self, pipe_no: int, entry: HistoryExportEntry) -> dict[str, Any]:
        summary_df = self._read_csv(entry.summary_csv)
        summary_row = self._find_summary_row(summary_df, serial=entry.serial, run_id=entry.run_id)
        if summary_row is None:
            raise RuntimeError(f"summary.csv 中找不到记录: {entry.serial}")

        section_df = self._filtered_section_df(
            self._read_csv(entry.section_results_csv),
            serial=entry.serial,
            run_id=entry.run_id,
        )

        row: dict[str, Any] = {
            "编号": int(pipe_no),
            "流水号": entry.serial,
            "平均外径": self._summary_or_section_mean(summary_row, section_df, "od_mean_mm", "od_avg_mm"),
            "外圆直线度": self._number_or_blank(self._series_value(summary_row, "od_end_off_mm")),
            "外径极差": self._number_or_blank(self._series_value(summary_row, "od_range_mm")),
            "平均内径": self._summary_or_section_mean(summary_row, section_df, "id_mean_mm", "id_avg_mm"),
            "内圆直线度": self._number_or_blank(self._series_value(summary_row, "id_end_off_mm")),
        }

        for section_idx in range(1, SECTION_COUNT + 1):
            row[f"截面{section_idx}外径真圆度"] = self._section_number(section_df, section_idx, "od_pp_rob_mm")
        for section_idx in range(1, SECTION_COUNT + 1):
            row[f"截面{section_idx}内径真圆度"] = self._section_number(section_df, section_idx, "id_round_fit_mm")
        for section_idx in range(1, SECTION_COUNT + 1):
            row[f"截面{section_idx}同心度"] = self._section_number(section_df, section_idx, "concentricity_mm")
        return row

    def _section_results_are_complete(self, path: Path, *, serial: str, run_id: str) -> bool:
        df = self._filtered_section_df(self._read_csv(path), serial=serial, run_id=run_id)
        if df.empty:
            return False
        if any(col not in df.columns for col in _REQUIRED_SECTION_COLUMNS):
            return False
        sections = self._section_indexes(df)
        if not all(index in sections for index in range(1, SECTION_COUNT + 1)):
            return False
        metric_columns = _REQUIRED_SECTION_COLUMNS - {"section_idx"}
        for section_idx in range(1, SECTION_COUNT + 1):
            for column in metric_columns:
                if self._section_number(df, section_idx, column) == "":
                    return False
        return True

    def _filtered_section_df(self, df: Any, *, serial: str, run_id: str) -> Any:
        out = df.copy()
        if run_id and "run_id" in out.columns:
            mask = out["run_id"].astype(str).str.strip() == str(run_id)
            if bool(mask.any()):
                return out.loc[mask].copy()
        if serial and "serial" in out.columns:
            mask = out["serial"].astype(str).str.strip() == str(serial)
            if bool(mask.any()):
                return out.loc[mask].copy()
        return out

    def _find_summary_row(self, df: Any, *, serial: str, run_id: str) -> Any | None:
        if run_id and "run_id" in df.columns:
            mask = df["run_id"].astype(str).str.strip() == str(run_id)
            matched = df.loc[mask]
            if not matched.empty:
                return matched.iloc[0]
        if serial and "serial" in df.columns:
            mask = df["serial"].astype(str).str.strip() == str(serial)
            matched = df.loc[mask]
            if not matched.empty:
                return matched.iloc[0]
        return None

    def _section_indexes(self, df: Any) -> set[int]:
        pd = self._pandas()
        values = pd.to_numeric(df["section_idx"], errors="coerce").dropna()
        return {int(v) for v in values.tolist()}

    def _section_number(self, df: Any, section_idx: int, column: str) -> float | str:
        if column not in df.columns or "section_idx" not in df.columns:
            return ""
        pd = self._pandas()
        idx = pd.to_numeric(df["section_idx"], errors="coerce")
        subset = df.loc[idx == int(section_idx)]
        if subset.empty:
            return ""
        values = pd.to_numeric(subset[column], errors="coerce").dropna()
        if values.empty:
            return ""
        return self._number_or_blank(values.iloc[0])

    def _summary_or_section_mean(self, summary_row: Any, section_df: Any, summary_col: str, section_col: str) -> float | str:
        val = self._number_or_blank(self._series_value(summary_row, summary_col))
        if val != "":
            return val
        if section_col not in section_df.columns:
            return ""
        pd = self._pandas()
        values = pd.to_numeric(section_df[section_col], errors="coerce").dropna()
        if values.empty:
            return ""
        return self._number_or_blank(values.mean())

    def _is_complete_meta(self, meta: dict[str, Any], status: str) -> bool:
        if self._truthy(meta.get("completed", None)):
            return True
        if "completed" not in meta and str(status or "").upper() == "DONE":
            return True
        return False

    def _display_start_time(self, date_text: str, summary_row: Any) -> str:
        start = str(self._series_value(summary_row, "start_time") or "").strip()
        if not start:
            return str(date_text)
        if str(date_text) and len(start) <= 8:
            return f"{date_text} {start}"
        return start

    def _parse_time_key(self, date_text: str, start_text: str) -> float:
        text = str(start_text or "").strip()
        candidates = [text]
        if len(text) <= 8 and date_text:
            candidates.insert(0, f"{date_text} {text}")
        for candidate in candidates:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                try:
                    return _dt.datetime.strptime(candidate[:19], fmt).timestamp()
                except Exception:
                    pass
        try:
            return _dt.datetime.strptime(str(date_text), "%Y-%m-%d").timestamp()
        except Exception:
            return 0.0

    def _series_value(self, row: Any, column: str) -> Any:
        try:
            if column in row.index:
                return row[column]
        except Exception:
            pass
        return ""

    def _number_or_blank(self, value: Any) -> float | str:
        if value in (None, ""):
            return ""
        try:
            val = float(value)
            if math.isnan(val):
                return ""
            return val
        except Exception:
            return ""

    def _truthy(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value or "").strip().lower()
        return text in {"1", "true", "yes", "y", "done"}

    def _read_csv(self, path: Path) -> Any:
        pd = self._pandas()
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
        missing_summary = _REQUIRED_SUMMARY_COLUMNS - set(df.columns)
        if path.name.lower().startswith("summary") and missing_summary:
            raise RuntimeError(f"{path.name} 缺少字段: {', '.join(sorted(missing_summary))}")
        return df

    def _pandas(self) -> Any:
        import pandas as pd  # type: ignore[import-not-found]

        return pd


__all__ = [
    "DETECTION_SUMMARY_COLUMNS",
    "HISTORY_INDEX_FILENAME",
    "HISTORY_INDEX_SCHEMA_VERSION",
    "HistoryExportEntry",
    "HistoryResultExportService",
]
