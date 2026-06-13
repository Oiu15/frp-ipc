from __future__ import annotations

"""Serial template rendering and counter-key helpers."""

import datetime as _dt
import re
from dataclasses import dataclass
from typing import Any, Mapping


SYSTEM_KEYS = {"date", "time", "recipe", "seq"}
CUSTOM_KEYS = {"customer", "batch", "work_order", "team", "remark"}
FIELD_TYPES = {"system", "custom", "text"}
DEFAULT_SEPARATOR = "-"
DEFAULT_SEQ_WIDTH = 3
MAX_PART_LEN = 24
MAX_TEXT_LEN = 48
WINDOWS_INVALID_CHARS = '<>:"/\\|?*'


@dataclass(frozen=True, slots=True)
class SerialGenerationResult:
    serial: str
    seq: int | None
    seq_text: str
    has_seq: bool
    suffix_added: bool
    counter_day: str
    counter_key: str | None
    warning: str


def default_serial_template() -> dict[str, Any]:
    return {
        "separator": DEFAULT_SEPARATOR,
        "custom_values": {
            "customer": "",
            "batch": "",
            "work_order": "",
            "team": "",
            "remark": "",
        },
        "fields": [
            {"type": "system", "key": "date", "enabled": True},
            {"type": "system", "key": "recipe", "enabled": True},
            {"type": "system", "key": "seq", "enabled": True},
        ],
    }


def normalize_serial_part(value: Any, *, max_len: int = MAX_PART_LEN, default: str = "") -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        return default
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if code < 32 or ch in WINDOWS_INVALID_CHARS:
            out.append("_")
        elif ch.isspace():
            out.append("_")
        else:
            out.append(ch)
    normalized = re.sub(r"_+", "_", "".join(out)).strip("_")
    if not normalized:
        return default
    return normalized[:max_len]


def normalize_separator(value: Any) -> str:
    sep = str(value if value is not None else DEFAULT_SEPARATOR)
    if sep in {"-", "_", " "}:
        return sep
    cleaned = normalize_serial_part(sep, max_len=8, default=DEFAULT_SEPARATOR)
    return cleaned if cleaned else DEFAULT_SEPARATOR


def normalize_serial_template(template: Mapping[str, Any] | None) -> dict[str, Any]:
    if template is None:
        return default_serial_template()
    if not isinstance(template, Mapping):
        raise ValueError("serial_template must be an object")

    raw_fields = template.get("fields")
    if not isinstance(raw_fields, list) or not raw_fields:
        raise ValueError("serial_template.fields must be a non-empty list")

    custom_values = dict(default_serial_template()["custom_values"])
    raw_custom_values = template.get("custom_values", {})
    if raw_custom_values is not None:
        if not isinstance(raw_custom_values, Mapping):
            raise ValueError("serial_template.custom_values must be an object")
        for key in CUSTOM_KEYS:
            custom_values[key] = normalize_serial_part(raw_custom_values.get(key, ""), max_len=MAX_PART_LEN)

    normalized_fields: list[dict[str, Any]] = []
    for raw in raw_fields:
        if not isinstance(raw, Mapping):
            raise ValueError("serial_template.fields items must be objects")
        field_type = str(raw.get("type", "") or "").strip()
        key = str(raw.get("key", "") or "").strip()
        enabled = bool(raw.get("enabled", True))
        if field_type not in FIELD_TYPES:
            raise ValueError(f"unsupported serial field type: {field_type}")
        if field_type == "system" and key not in SYSTEM_KEYS:
            raise ValueError(f"unsupported serial system field: {key}")
        if field_type == "custom" and key not in CUSTOM_KEYS:
            raise ValueError(f"unsupported serial custom field: {key}")
        if field_type == "text":
            key = "text"
        field: dict[str, Any] = {"type": field_type, "key": key, "enabled": enabled}
        if field_type == "text":
            field["value"] = normalize_serial_part(raw.get("value", ""), max_len=MAX_TEXT_LEN)
        normalized_fields.append(field)

    return {
        "separator": normalize_separator(template.get("separator", DEFAULT_SEPARATOR)),
        "custom_values": custom_values,
        "fields": normalized_fields,
    }


def validate_serial_template(template: Mapping[str, Any] | None) -> list[str]:
    normalized = normalize_serial_template(template)
    warnings: list[str] = []
    if not any(_is_enabled_seq(field) for field in normalized["fields"]):
        warnings.append("当前模板不包含序号字段，短码会显示在流水号、导出目录和报表中。")
    if not any(bool(field.get("enabled", True)) for field in normalized["fields"]):
        raise ValueError("serial_template must contain at least one enabled field")
    return warnings


def build_preview_serial(
    template: Mapping[str, Any] | None,
    *,
    recipe_name: str,
    now: _dt.datetime | None = None,
    run_id: str = "A4F91C00000000000000000000000000",
) -> SerialGenerationResult:
    return _render_serial(
        normalize_serial_template(template),
        recipe_name=recipe_name,
        now=now,
        run_id=run_id,
        seq=1,
        update_counter=False,
    )


def generate_serial(
    template: Mapping[str, Any] | None,
    *,
    recipe_name: str,
    run_id: str,
    counters: dict[str, Any],
    now: _dt.datetime | None = None,
) -> SerialGenerationResult:
    normalized = normalize_serial_template(template)
    day_tag = _day_tag(now)
    counter_key = build_counter_key(normalized, recipe_name=recipe_name, now=now)
    has_seq = any(_is_enabled_seq(field) for field in normalized["fields"])
    seq = None
    if has_seq:
        day_map = counters.get(day_tag, {})
        if not isinstance(day_map, dict):
            day_map = {}
        try:
            seq = int(day_map.get(counter_key, 0)) + 1
        except Exception:
            seq = 1
        day_map[counter_key] = seq
        counters[day_tag] = day_map
    return _render_serial(
        normalized,
        recipe_name=recipe_name,
        now=now,
        run_id=run_id,
        seq=seq or 1,
        update_counter=True,
    )


def build_counter_key(
    template: Mapping[str, Any] | None,
    *,
    recipe_name: str,
    now: _dt.datetime | None = None,
) -> str:
    normalized = normalize_serial_template(template)
    parts: list[str] = []
    for field in normalized["fields"]:
        if not bool(field.get("enabled", True)):
            continue
        field_type = str(field.get("type", ""))
        key = str(field.get("key", ""))
        if field_type == "system" and key == "recipe":
            value = normalize_serial_part(recipe_name, default="recipe")
        elif field_type == "custom" and key in CUSTOM_KEYS:
            value = normalize_serial_part(normalized["custom_values"].get(key, ""))
        else:
            continue
        if value:
            parts.append(value)
    return "|".join(parts) if parts else _day_tag(now)


def _render_serial(
    template: Mapping[str, Any],
    *,
    recipe_name: str,
    now: _dt.datetime | None,
    run_id: str,
    seq: int,
    update_counter: bool,
) -> SerialGenerationResult:
    current = now or _dt.datetime.now()
    day_tag = current.strftime("%Y%m%d")
    values = {
        "date": day_tag,
        "time": current.strftime("%H%M%S"),
        "recipe": normalize_serial_part(recipe_name, default="recipe"),
        "seq": f"{max(1, int(seq)):0{DEFAULT_SEQ_WIDTH}d}",
    }
    fields = list(template["fields"])
    separator = str(template["separator"])
    parts: list[str] = []
    has_seq = False
    for field in fields:
        if not bool(field.get("enabled", True)):
            continue
        field_type = str(field.get("type", ""))
        key = str(field.get("key", ""))
        value = ""
        if field_type == "system":
            if key == "seq":
                has_seq = True
            value = values.get(key, "")
        elif field_type == "custom":
            value = normalize_serial_part(template["custom_values"].get(key, ""))
        elif field_type == "text":
            value = normalize_serial_part(field.get("value", ""), max_len=MAX_TEXT_LEN)
        if value:
            parts.append(value)
    serial = separator.join(parts) if parts else day_tag
    suffix_added = False
    warning = ""
    if not has_seq:
        suffix = str(run_id or "")[:6].upper()
        suffix = normalize_serial_part(suffix, max_len=6, default="RUNID")[:6]
        serial = f"{serial}__{suffix}"
        suffix_added = True
        warning = "当前模板不包含序号字段，短码会显示在流水号、导出目录和报表中。"
    counter_key = build_counter_key(template, recipe_name=recipe_name, now=now) if has_seq else None
    return SerialGenerationResult(
        serial=serial,
        seq=(int(seq) if has_seq else None),
        seq_text=(f"{max(1, int(seq)):0{DEFAULT_SEQ_WIDTH}d}" if has_seq else ""),
        has_seq=has_seq,
        suffix_added=suffix_added,
        counter_day=day_tag,
        counter_key=counter_key,
        warning=warning,
    )


def _day_tag(now: _dt.datetime | None) -> str:
    return (now or _dt.datetime.now()).strftime("%Y%m%d")


def _is_enabled_seq(field: Mapping[str, Any]) -> bool:
    return (
        bool(field.get("enabled", True))
        and str(field.get("type", "")) == "system"
        and str(field.get("key", "")) == "seq"
    )


__all__ = [
    "CUSTOM_KEYS",
    "SerialGenerationResult",
    "SYSTEM_KEYS",
    "build_counter_key",
    "build_preview_serial",
    "default_serial_template",
    "generate_serial",
    "normalize_serial_part",
    "normalize_serial_template",
    "validate_serial_template",
]
