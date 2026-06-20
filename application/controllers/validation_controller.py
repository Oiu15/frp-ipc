from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from domain.state import (
    FIXED_SECTION_PRIMARY_METRICS,
    VALIDATION_MOVE_CHANNELS,
    VALIDATION_MOVE_SCENARIOS,
)


class ValidationHostPort(Protocol):
    def list_validation_section_choices(self) -> list[str]: ...

    def start_validation_run(self, **kwargs: Any) -> Any: ...

    def stop_validation_run(self) -> Any: ...


def _coerce_bool(value: bool | str | int) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"", "0", "false", "no", "n", "off"}:
        return False
    return bool(value)


def _coerce_non_negative_float(value: str | int | float, field_name: str) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        numeric = float(text)
    except Exception as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if numeric < 0.0:
        raise ValueError(f"{field_name} must be >= 0")
    return numeric


def _coerce_positive_float(value: str | int | float, field_name: str) -> float:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be > 0")
    try:
        numeric = float(text)
    except Exception as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if numeric <= 0.0:
        raise ValueError(f"{field_name} must be > 0")
    return numeric


def _coerce_choice(value: str, field_name: str, choices: Sequence[str]) -> str:
    text = str(value or "").strip()
    if text not in choices:
        raise ValueError(f"{field_name} must be one of: " + ", ".join(choices))
    return text


def _coerce_positive_int(value: str | int | float, field_name: str) -> int:
    text = str(value or "").strip()
    if ":" in text:
        text = text.split(":", 1)[0].strip()
    if not text:
        raise ValueError(f"{field_name} must be >= 1")
    try:
        numeric = int(float(text))
    except Exception as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if numeric < 1:
        raise ValueError(f"{field_name} must be >= 1")
    return numeric


class ValidationController:
    def __init__(self, host: ValidationHostPort) -> None:
        self._host = host

    def list_validation_section_choices(self) -> list[str]:
        try:
            values = self._host.list_validation_section_choices()
            if values:
                return list(values)
        except Exception:
            pass
        return ["1"]

    def start_validation_run(
        self,
        section_name: str,
        metric_name: str,
        repeat_count: str | int,
        reclamp_between_repeats: bool | str | int = False,
        *,
        reclamp_enabled: bool | str | int = False,
        rotation_stop_before_measure: bool | str | int = False,
        release_settle_s: str | int | float = 0.0,
        clamp_settle_s: str | int | float = 0.0,
        position_settle_s: str | int | float = 0.0,
        sample_delay_s: str | int | float = 0.0,
        validation_ax3_speed_dps: str | int | float = 60.0,
        move_enabled: bool | str | int = False,
        move_channel: str = "od_channel",
        move_away_delta_mm: str | int | float = 0.0,
        move_scenario: str = "distance_round_trip",
        move_from_section_index: str | int | float = 1,
        move_target_section_index: str | int | float = 1,
        move_return_section_index: str | int | float = 1,
    ) -> Any:
        try:
            section = str(section_name or "").strip()
            metric = str(metric_name or "").strip()
            repeat_raw = str(repeat_count).strip()
            if not repeat_raw:
                raise ValueError("repeat_count cannot be empty")
            try:
                repeat = int(repeat_raw)
            except Exception as exc:
                raise ValueError("repeat_count must be a positive integer") from exc
            if repeat < 1:
                raise ValueError("repeat_count must be >= 1")
            if metric not in FIXED_SECTION_PRIMARY_METRICS:
                raise ValueError(
                    "metric_name must be one of: " + ", ".join(FIXED_SECTION_PRIMARY_METRICS)
                )
            return self._host.start_validation_run(
                section_name=section,
                metric_name=metric,
                repeat_count=repeat,
                reclamp_between_repeats=_coerce_bool(reclamp_between_repeats),
                reclamp_enabled=_coerce_bool(reclamp_enabled),
                rotation_stop_before_measure=_coerce_bool(rotation_stop_before_measure),
                release_settle_s=_coerce_non_negative_float(release_settle_s, "release_settle_s"),
                clamp_settle_s=_coerce_non_negative_float(clamp_settle_s, "clamp_settle_s"),
                position_settle_s=_coerce_non_negative_float(position_settle_s, "position_settle_s"),
                sample_delay_s=_coerce_non_negative_float(sample_delay_s, "sample_delay_s"),
                validation_ax3_speed_dps=_coerce_positive_float(
                    validation_ax3_speed_dps,
                    "validation_ax3_speed_dps",
                ),
                move_enabled=_coerce_bool(move_enabled),
                move_channel=_coerce_choice(
                    move_channel,
                    "move_channel",
                    VALIDATION_MOVE_CHANNELS,
                ),
                move_away_delta_mm=_coerce_non_negative_float(
                    move_away_delta_mm,
                    "move_away_delta_mm",
                ),
                move_scenario=_coerce_choice(
                    move_scenario,
                    "move_scenario",
                    VALIDATION_MOVE_SCENARIOS,
                ),
                move_from_section_index=_coerce_positive_int(
                    move_from_section_index,
                    "move_from_section_index",
                ),
                move_target_section_index=_coerce_positive_int(
                    move_target_section_index,
                    "move_target_section_index",
                ),
                move_return_section_index=_coerce_positive_int(
                    move_return_section_index,
                    "move_return_section_index",
                ),
            )
        except Exception as exc:
            setter = getattr(self._host, "_set_validation_feedback", None)
            if not callable(setter):
                setter = getattr(self._host, "_set_validation_debug_feedback", None)
            if callable(setter):
                setter(status="ERR", result="", error=str(exc), export_path="")
            return None

    def stop_validation_run(self) -> Any:
        return self._host.stop_validation_run()


__all__ = ["ValidationController", "ValidationHostPort"]
