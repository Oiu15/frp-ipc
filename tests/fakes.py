from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, Sequence

from domain.state import RunContext, RunIdentity, ValidationExportContext
from core.models import AxisComm
from machine.device_gateway import ClChannel, ClReadResult, PollProfile, RegsRead


# ---------------------------------------------------------------------------
# Lightweight UI stand-ins shared across many test modules
# ---------------------------------------------------------------------------


class FakeVar:
    """Minimal Tkinter StringVar / IntVar stand-in."""

    def __init__(self, value: object = "") -> None:
        self.value = value

    def get(self) -> object:
        return self.value

    def set(self, value: object) -> None:
        self.value = value


class FakeCombo:
    """Minimal ttk.Combobox stand-in used by gauge-connection tests."""

    def __init__(self, value: str = "") -> None:
        self.value = value
        self.configs: list[dict[str, Any]] = []

    def configure(self, **kwargs: Any) -> None:
        self.configs.append(dict(kwargs))

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


@dataclass
class FakeWidget:
    """Minimal Tk widget stand-in that records state changes."""

    def __init__(self) -> None:
        self.states: list[str] = []

    def configure(self, **kwargs: Any) -> None:
        self.states.append(str(kwargs.get("state", "")))


# ---------------------------------------------------------------------------
# Core test doubles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GatewayCall:
    name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass
class _GatewayAllowance:
    return_value: Any = None
    remaining: int | None = None


class StrictDeviceGateway:
    """Device gateway fake that only permits explicitly allowed calls."""

    def __init__(self) -> None:
        self.calls: list[GatewayCall] = []
        self._allowances: dict[str, _GatewayAllowance] = {}

    def allow(self, name: str, *, return_value: Any = None, times: int | None = None) -> StrictDeviceGateway:
        if times is not None and times < 0:
            raise ValueError("times must be >= 0")
        self._allowances[str(name)] = _GatewayAllowance(return_value=return_value, remaining=times)
        return self

    def assert_called(self, name: str, *, times: int | None = None) -> None:
        count = sum(1 for call in self.calls if call.name == name)
        if times is None:
            if count == 0:
                raise AssertionError(f"expected gateway call: {name}")
            return
        if count != times:
            raise AssertionError(f"expected gateway call {name!r} {times} time(s), got {count}")

    def _unexpected_gateway_call(self, name: str) -> NoReturn:
        raise AssertionError(f"unexpected gateway call: {name}")

    def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        allowance = self._allowances.get(name)
        if allowance is None:
            self._unexpected_gateway_call(name)
        if allowance.remaining is not None:
            if allowance.remaining <= 0:
                self._unexpected_gateway_call(name)
            allowance.remaining -= 1
        self.calls.append(GatewayCall(name=name, args=args, kwargs=dict(kwargs)))
        return allowance.return_value

    def __getattr__(self, name: str) -> Any:
        self._unexpected_gateway_call(name)

    def get_axis_copy(self, axis: int) -> AxisComm:
        return self._call("get_axis_copy", axis)

    def movea_abs(self, axis: int, pos_abs: float, *, context: str = "MoveA") -> None:
        self._call("movea_abs", axis, pos_abs, context=context)

    def velmove(
        self,
        axis: int,
        velocity: float,
        *,
        acc: float = 80.0,
        dec: float = 80.0,
        jerk: float = 300.0,
    ) -> None:
        self._call("velmove", axis, velocity, acc=acc, dec=dec, jerk=jerk)

    def stop(self, axis: int) -> None:
        self._call("stop", axis)

    def halt(self, axis: int) -> None:
        self._call("halt", axis)

    def reset(self, axis: int) -> None:
        self._call("reset", axis)

    def enable(self, axis: int) -> None:
        self._call("enable", axis)

    def abort_motion(self, axes: Sequence[int] | None = None) -> None:
        self._call("abort_motion", axes)

    def apply_soft_limits_abs(
        self,
        axis: int,
        target_abs: float,
        *,
        strict: bool = False,
        context: str = "",
    ) -> float:
        return self._call("apply_soft_limits_abs", axis, target_abs, strict=strict, context=context)

    def read_regs_sync(self, d_addr: int, count: int, timeout_s: float = 0.35) -> RegsRead | None:
        return self._call("read_regs_sync", d_addr, count, timeout_s)

    def read_axis_angle_deg_sync(self, axis: int = 3, timeout_s: float = 0.35) -> float | None:
        return self._call("read_axis_angle_deg_sync", axis, timeout_s)

    def read_cl_sync(self, channel: ClChannel, *, timeout_s: float = 0.5) -> ClReadResult | None:
        return self._call("read_cl_sync", channel, timeout_s=timeout_s)

    def set_plc_poll_profile(self, profile: PollProfile = "normal") -> None:
        self._call("set_plc_poll_profile", profile)

    def pulse_cmd_mask(self, axis: int, pulse_mask: int, pulse_ms: int = 120) -> None:
        self._call("pulse_cmd_mask", axis, pulse_mask, pulse_ms)

    def write_coil(self, coil_addr: int, value: int | bool) -> None:
        self._call("write_coil", coil_addr, value)


class SequentialRunRepository:
    def __init__(
        self,
        *,
        serial_prefix: str = "20260408-validation",
        run_id_prefix: str = "run",
        started_at_base_ts: float = 1775606400.0,
    ) -> None:
        self._seq = 0
        self.serial_prefix = serial_prefix
        self.run_id_prefix = run_id_prefix
        self.started_at_base_ts = float(started_at_base_ts)
        self.prepared: list[tuple[str, str]] = []

    def prepare_run(self, recipe_name: str) -> RunIdentity:
        self._seq += 1
        serial = f"{self.serial_prefix}-{self._seq:03d}"
        run_id = f"{self.run_id_prefix}-{self._seq:03d}"
        started_at_ts = self.started_at_base_ts + float(self._seq)
        self.prepared.append((recipe_name, serial))
        return RunIdentity(serial=serial, run_id=run_id, started_at_ts=started_at_ts)

    def export_run(self, context: RunContext) -> str:
        raise AssertionError("unexpected repository call: export_run")

    def export_daily_summary(self, context: RunContext) -> None:
        raise AssertionError("unexpected repository call: export_daily_summary")


class RecordingValidationRepository:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.exported_run_paths: list[Path] = []
        self.exported_summary_paths: list[Path] = []
        self.exported_statuses: list[str] = []
        self.exported_contexts: list[ValidationExportContext] = []

    def export_run(self, context: ValidationExportContext) -> str:
        run_dir = self._run_dir(context)
        self.exported_contexts.append(context)
        self.exported_run_paths.append(run_dir)
        self.exported_statuses.append(str(context.status))
        self.export_daily_summary(context)
        return str(run_dir)

    def export_fixed_section_repeatability(self, **_kwargs: Any) -> str:
        raise AssertionError("unexpected repository call: export_fixed_section_repeatability")

    def export_daily_summary(self, context: ValidationExportContext) -> None:
        start_ts = self._start_ts(context)
        day_tag = dt.date.fromtimestamp(start_ts).strftime("%Y-%m-%d")
        self.exported_summary_paths.append(self.root / "validation_exports" / day_tag / "summary.csv")

    def _run_dir(self, context: ValidationExportContext) -> Path:
        start_ts = self._start_ts(context)
        day_tag = dt.date.fromtimestamp(start_ts).strftime("%Y-%m-%d")
        return self.root / "validation_exports" / day_tag / str(context.identity.serial)

    def _start_ts(self, context: ValidationExportContext) -> float:
        if context.started_at_ts is not None:
            return float(context.started_at_ts)
        return float(context.identity.started_at_ts)
