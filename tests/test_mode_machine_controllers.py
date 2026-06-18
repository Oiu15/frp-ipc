from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from application.adapters.calibration_view import AppCalibrationViewAdapter
from services.calibration_controller import CalibrationController
from services.id_calibration import IdCalibrationService
from services.id_single_calibration import IdSingleCalibrationService
from services.measurement_service import MeasurementController
from modes.mode_machine import ModeMachine
from services.od_calibration import OdCalibrationService


class _FakeMode:
    def __init__(self) -> None:
        self.start_calls = 0
        self.state_name = "idle"
        self.last_error = None

    def start(self):
        self.start_calls += 1
        self.state_name = "preparing"
        return "started"


class _FakeRuntimeState:
    def __init__(self) -> None:
        self.mode_kind = "none"
        self.mode_state = "idle"
        self.mode_error = None


class _FakeModeMachine:
    def __init__(self) -> None:
        self.entered: list[str] = []
        self.current_mode = None
        self.stop_calls = 0
        self.sync_calls = 0
        self.runtime_state = _FakeRuntimeState()

    def enter_production(self):
        self.entered.append("production")
        self.runtime_state.mode_kind = "production"
        self.current_mode = _FakeMode()
        return self.current_mode

    def enter_calibration(self):
        self.entered.append("calibration")
        self.runtime_state.mode_kind = "calibration"
        return object()

    def stop_current(self):
        self.stop_calls += 1
        self.runtime_state.mode_state = "idle"
        return "stopped"

    def sync_current_mode_state(self):
        self.sync_calls += 1
        if self.current_mode is not None:
            self.runtime_state.mode_state = self.current_mode.state_name
            self.runtime_state.mode_error = self.current_mode.last_error


class _FakeVar:
    def __init__(self, value="") -> None:
        self.value = value

    def get(self):
        return self.value

    def set(self, value) -> None:
        self.value = value


class _FakeCalibrationHost:
    def __init__(self) -> None:
        self.odcal_rot_degps_var = _FakeVar("11")
        self.odcal_hz_var = _FakeVar("12")
        self.odcal_duration_var = _FakeVar("13")
        self.odcal_dref_var = _FakeVar("181")
        self.odcal_mode_var = _FakeVar("timed")
        self.odcal_angle_src_var = _FakeVar("AX3")
        self.odcal_filter_var = _FakeVar("median")
        self.odcal_outlier_sigma_var = _FakeVar("2.5")
        self.odcal_gauge_cmd_var = _FakeVar("M0,1")
        self.odcal_map_out1_var = _FakeVar("L")
        self.odcal_state_var = _FakeVar()
        self.odcal_msg_var = _FakeVar()
        self.odcal_B_candidate_var = _FakeVar()
        self.odcal_B_active_var = _FakeVar()
        self.odcal_n_var = _FakeVar()

        self.idcal_rot_degps_var = _FakeVar("21")
        self.idcal_hz_var = _FakeVar("22")
        self.idcal_duration_var = _FakeVar("23")
        self.idcal_mode_var = _FakeVar("timed")
        self.idcal_dref_var = _FakeVar("151")
        self.idcal_delta_candidate_var = _FakeVar()
        self.idcal_delta_active_var = _FakeVar()
        self.idcal_state_var = _FakeVar()
        self.idcal_msg_var = _FakeVar()
        self.idcal_chk_err_var = _FakeVar()
        self.idcal_chk_cov_var = _FakeVar()
        self.idcal_chk_n_var = _FakeVar()
        self.idcal_chk_dtheta_var = _FakeVar()

        self.id_single_cal_dref_var = _FakeVar("152")
        self.id_single_cal_state_var = _FakeVar()
        self.id_single_cal_msg_var = _FakeVar()
        self.id_single_cal_mean_var = _FakeVar()
        self.id_single_cal_B_var = _FakeVar()
        self.id_single_cal_cov_var = _FakeVar()

    def _parse_float(self, value, default):
        try:
            return float(value)
        except Exception:
            return default


class _FakeOdService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.export_result: dict[str, Any] = {"ok": True, "path": Path("od_raw.csv"), "n": 1}

    def start_capture(self, *args, **kwargs) -> None:
        self.calls.append(("start_capture", args, kwargs))

    def compute_candidate(self, *args, **kwargs):
        self.calls.append(("compute_candidate", args, kwargs))
        return {"ok": True, "b_mm": 1.25, "n": 8}

    def export_raw(self):
        self.calls.append(("export_raw", (), {}))
        return self.export_result


class _FakeIdService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.export_result: dict[str, Any] = {"ok": True, "path": Path("id_raw.csv"), "n": 1}
        self.active: dict[str, Any] = {"delta_c_mm": 0.12, "D_ref": 151.0}

    def start_capture(self, *args, **kwargs) -> None:
        self.calls.append(("start_capture", args, kwargs))

    def compute_candidate(self, *args, **kwargs):
        self.calls.append(("compute_candidate", args, kwargs))
        return {"ok": True, "delta_c_mm": 0.12}

    def export_raw(self):
        self.calls.append(("export_raw", (), {}))
        return self.export_result

    def load_active(self):
        self.calls.append(("load_active", (), {}))
        return dict(self.active)

    def start_verify_capture(self, *args, **kwargs) -> None:
        self.calls.append(("start_verify_capture", args, kwargs))


class _FakeIdSingleService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def start_capture(self, *args, **kwargs) -> None:
        self.calls.append(("start_capture", args, kwargs))

    def compute_and_apply(self, *args, **kwargs):
        self.calls.append(("compute_and_apply", args, kwargs))
        return {"ok": True, "mean_l2_mm": 10.0, "b_mm": 0.5, "cov_pct": 98.0}


class _FakeCalibrationView:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {
            "odcal_rot_degps_var": "11",
            "odcal_hz_var": "12",
            "odcal_duration_var": "13",
            "odcal_dref_var": "181",
            "odcal_mode_var": "timed",
            "odcal_angle_src_var": "AX3",
            "odcal_filter_var": "median",
            "odcal_outlier_sigma_var": "2.5",
            "odcal_cmd_var": "M0,1",
            "idcal_rot_degps_var": "21",
            "idcal_hz_var": "22",
            "idcal_duration_var": "23",
            "idcal_mode_var": "timed",
            "idcal_dref_var": "151",
            "idcal_delta_active_var": "0.22",
        }
        self.reads: list[str] = []
        self.writes: list[tuple[str, Any]] = []

    def get_value(self, name: str, default: Any = None) -> Any:
        self.reads.append(name)
        return self.values.get(name, default)

    def set_value(self, name: str, value: Any) -> None:
        self.writes.append((name, value))
        self.values[name] = value

    def get_float(self, name: str, default: float) -> float:
        self.reads.append(name)
        try:
            return float(self.values.get(name, default))
        except Exception:
            return float(default)


class TestAppCalibrationViewAdapter:
    def test_reads_and_writes_host_tk_like_vars(self) -> None:
        host = _FakeCalibrationHost()
        adapter = AppCalibrationViewAdapter(host)

        assert adapter.get_value("idcal_hz_var") == "22"
        adapter.set_value("idcal_hz_var", "44")

        assert host.idcal_hz_var.get() == "44"

    def test_missing_var_uses_default_and_write_is_noop(self) -> None:
        adapter = AppCalibrationViewAdapter(_FakeCalibrationHost())

        assert adapter.get_value("missing_var", "fallback") == "fallback"
        adapter.set_value("missing_var", "ignored")

    def test_get_float_uses_host_parser_when_available(self) -> None:
        host = _FakeCalibrationHost()
        host.idcal_hz_var.set("bad")
        adapter = AppCalibrationViewAdapter(host)

        assert adapter.get_float("idcal_hz_var", 20.0) == 20.0

    def test_get_float_without_parser_falls_back_to_float_conversion(self) -> None:
        class HostWithoutParser:
            value_var = _FakeVar("12.5")

        adapter = AppCalibrationViewAdapter(HostWithoutParser())

        assert adapter.get_float("value_var", 1.0) == 12.5
        assert adapter.get_float("missing_var", 1.0) == 1.0


class TestControllerModeMachine:
    def test_measurement_controller_uses_mode_machine(self) -> None:
        machine = _FakeModeMachine()
        controller = MeasurementController(mode_machine=cast(ModeMachine, machine))

        assert controller.start_measurement() == "started"
        assert machine.entered == ["production"]
        assert machine.current_mode is not None
        current_mode = cast(_FakeMode, machine.current_mode)
        assert current_mode.start_calls == 1
        assert machine.sync_calls == 1
        assert machine.runtime_state.mode_kind == "production"
        assert machine.runtime_state.mode_state == "preparing"
        assert controller.stop_measurement() == "stopped"
        assert machine.stop_calls == 1
        assert machine.sync_calls == 2

    def test_calibration_controller_enters_calibration_before_service_call(self) -> None:
        machine = _FakeModeMachine()
        id_service = _FakeIdService()
        controller = CalibrationController(
            mode_machine=cast(ModeMachine, machine),
            view=_FakeCalibrationView(),
            id_service=cast(IdCalibrationService, id_service),
        )

        controller.compute_id_calibration()

        assert machine.entered == ["calibration"]
        assert machine.sync_calls == 1
        assert machine.runtime_state.mode_kind == "calibration"
        assert id_service.calls == [("compute_candidate", (151.0,), {})]

    def test_od_entrypoint_uses_injected_port_service(self) -> None:
        machine = _FakeModeMachine()
        od_service = _FakeOdService()
        host = _FakeCalibrationHost()
        controller = CalibrationController(
            mode_machine=cast(ModeMachine, machine),
            view=AppCalibrationViewAdapter(host),
            od_service=cast(OdCalibrationService, od_service),
        )

        controller.start_od_b_capture()
        controller.compute_od_b()

        assert od_service.calls[0][0] == "start_capture"
        assert od_service.calls[0][2]["sampling_hz"] == 12.0
        assert od_service.calls[1] == ("compute_candidate", (181.0, 2.5), {})
        assert host.odcal_B_candidate_var.get() == "1.25000"

    def test_id_entrypoint_uses_injected_port_service(self) -> None:
        machine = _FakeModeMachine()
        id_service = _FakeIdService()
        host = _FakeCalibrationHost()
        controller = CalibrationController(
            mode_machine=cast(ModeMachine, machine),
            view=AppCalibrationViewAdapter(host),
            id_service=cast(IdCalibrationService, id_service),
        )

        controller.start_id_capture()
        controller.compute_id_calibration()

        assert id_service.calls[0][0] == "start_capture"
        assert id_service.calls[0][2]["sampling_hz"] == 22.0
        assert id_service.calls[1] == ("compute_candidate", (151.0,), {})
        assert host.idcal_delta_candidate_var.get() == "0.1200"

    def test_id_single_entrypoint_uses_injected_port_service(self) -> None:
        machine = _FakeModeMachine()
        id_single_service = _FakeIdSingleService()
        host = _FakeCalibrationHost()
        controller = CalibrationController(
            mode_machine=cast(ModeMachine, machine),
            view=AppCalibrationViewAdapter(host),
            id_single_service=cast(IdSingleCalibrationService, id_single_service),
        )

        controller.start_id_single_capture()
        controller.compute_and_write_id_single_calibration()

        assert id_single_service.calls[0][0] == "start_capture"
        assert id_single_service.calls[0][2]["sampling_hz"] == 22.0
        assert id_single_service.calls[1] == ("compute_and_apply", (152.0,), {})
        assert host.id_single_cal_B_var.get() == "0.50000"

    def test_controller_requires_explicit_calibration_view(self) -> None:
        with pytest.raises(TypeError):
            CalibrationController(
                mode_machine=cast(ModeMachine, _FakeModeMachine()),
            )

    def test_missing_port_service_raises_clear_error(self) -> None:
        controller = CalibrationController(
            mode_machine=cast(ModeMachine, _FakeModeMachine()),
            view=_FakeCalibrationView(),
        )

        with pytest.raises(RuntimeError, match="OdCalibrationService not injected"):
            controller.start_od_b_capture()

    def test_od_entrypoint_reads_and_writes_through_view_port(self) -> None:
        machine = _FakeModeMachine()
        od_service = _FakeOdService()
        view = _FakeCalibrationView()
        controller = CalibrationController(
            mode_machine=cast(ModeMachine, machine),
            view=cast(Any, view),
            od_service=cast(OdCalibrationService, od_service),
        )

        controller.start_od_b_capture()
        controller.compute_od_b()

        assert "odcal_hz_var" in view.reads
        assert ("odcal_state_var", "CAPTURING") in view.writes
        assert ("odcal_B_candidate_var", "1.25000") in view.writes

    def test_od_raw_export_uses_injected_port_service(self) -> None:
        machine = _FakeModeMachine()
        od_service = _FakeOdService()
        host = _FakeCalibrationHost()
        controller = CalibrationController(
            mode_machine=cast(ModeMachine, machine),
            view=AppCalibrationViewAdapter(host),
            od_service=cast(OdCalibrationService, od_service),
        )

        controller.export_od_b_raw()

        assert od_service.calls == [("export_raw", (), {})]
        assert host.odcal_msg_var.get() == "已导出: od_raw.csv"

    def test_id_raw_export_uses_injected_port_service(self) -> None:
        machine = _FakeModeMachine()
        id_service = _FakeIdService()
        host = _FakeCalibrationHost()
        controller = CalibrationController(
            mode_machine=cast(ModeMachine, machine),
            view=AppCalibrationViewAdapter(host),
            id_service=cast(IdCalibrationService, id_service),
        )

        controller.export_id_raw()

        assert id_service.calls == [("export_raw", (), {})]
        assert host.idcal_msg_var.get() == "已导出: id_raw.csv"

    def test_verify_id_calibration_uses_injected_port_service(self) -> None:
        machine = _FakeModeMachine()
        id_service = _FakeIdService()
        host = _FakeCalibrationHost()
        controller = CalibrationController(
            mode_machine=cast(ModeMachine, machine),
            view=AppCalibrationViewAdapter(host),
            id_service=cast(IdCalibrationService, id_service),
        )

        controller.verify_id_calibration()

        assert id_service.calls[0] == ("load_active", (), {})
        name, args, kwargs = id_service.calls[1]
        assert name == "start_verify_capture"
        assert args == ()
        assert kwargs == {
            "rotation_speed_dps": 21.0,
            "sampling_hz": 22.0,
            "capture_duration_s": 23.0,
            "delta_c_mm": 0.12,
            "reference_diameter_mm": 151.0,
        }
        assert host.idcal_chk_err_var.get() == "--"
        assert host.idcal_chk_cov_var.get() == "--"
        assert host.idcal_chk_n_var.get() == "--"
        assert host.idcal_chk_dtheta_var.get() == "--"
        assert host.idcal_state_var.get() == "CHK"
        assert host.idcal_msg_var.get() == "复核采集中..."

    def test_verify_id_calibration_falls_back_to_view_active_delta(self) -> None:
        machine = _FakeModeMachine()
        id_service = _FakeIdService()
        id_service.active = {}
        view = _FakeCalibrationView()
        controller = CalibrationController(
            mode_machine=cast(ModeMachine, machine),
            view=cast(Any, view),
            id_service=cast(IdCalibrationService, id_service),
        )

        controller.verify_id_calibration()

        assert ("idcal_state_var", "CHK") in view.writes
        assert ("idcal_msg_var", "复核采集中...") in view.writes
        name, _args, kwargs = id_service.calls[-1]
        assert name == "start_verify_capture"
        assert kwargs["delta_c_mm"] == 0.22
        assert kwargs["reference_diameter_mm"] == 151.0

    def test_verify_id_calibration_missing_active_delta_sets_existing_error_text(self) -> None:
        machine = _FakeModeMachine()
        id_service = _FakeIdService()
        id_service.active = {}
        host = _FakeCalibrationHost()
        controller = CalibrationController(
            mode_machine=cast(ModeMachine, machine),
            view=AppCalibrationViewAdapter(host),
            id_service=cast(IdCalibrationService, id_service),
        )

        controller.verify_id_calibration()

        assert id_service.calls == [("load_active", (), {})]
        assert host.idcal_state_var.get() == "ERR"
        assert host.idcal_msg_var.get() == "复核失败：未找到 δc_active（请先“应用”）"

    def test_raw_export_no_data_sets_existing_ui_error_text(self) -> None:
        machine = _FakeModeMachine()
        od_service = _FakeOdService()
        od_service.export_result = {"ok": False, "reason": "无数据", "n": 0}
        host = _FakeCalibrationHost()
        controller = CalibrationController(
            mode_machine=cast(ModeMachine, machine),
            view=AppCalibrationViewAdapter(host),
            od_service=cast(OdCalibrationService, od_service),
        )

        controller.export_od_b_raw()

        assert host.odcal_state_var.get() == "ERR"
        assert host.odcal_msg_var.get() == "无数据"

    def test_raw_export_repository_error_sets_existing_ui_error_text(self) -> None:
        machine = _FakeModeMachine()
        id_service = _FakeIdService()
        id_service.export_result = {"ok": False, "reason": "导出失败: disk full", "n": 1}
        host = _FakeCalibrationHost()
        controller = CalibrationController(
            mode_machine=cast(ModeMachine, machine),
            view=AppCalibrationViewAdapter(host),
            id_service=cast(IdCalibrationService, id_service),
        )

        controller.export_id_raw()

        assert host.idcal_state_var.get() == "ERR"
        assert host.idcal_msg_var.get() == "导出失败: disk full"


def test_new_calibration_services_do_not_accept_legacy_host_any() -> None:
    root = Path(__file__).resolve().parents[1]
    service_paths = [
        root / "services" / "od_calibration.py",
        root / "services" / "id_calibration.py",
        root / "services" / "id_single_calibration.py",
    ]

    offenders: list[str] = []
    for path in service_paths:
        source = path.read_text(encoding="utf-8-sig")
        for token in ("host: Any", "tk.", "StringVar", "BooleanVar", "IntVar"):
            if token in source:
                offenders.append(f"{path.name}: {token}")

    assert offenders == []


def test_calibration_controller_has_no_legacy_service_fallback() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "services" / "calibration_controller.py").read_text(encoding="utf-8-sig")

    forbidden = [
        "from services.calibration_service import CalibrationService",
        "service: CalibrationService",
        "self.service",
        "_run_legacy_host_service",
        "host: Any",
        "HostCalibrationViewAdapter",
        "getattr(self.host",
        "_parse_float",
    ]

    assert [token for token in forbidden if token in source] == []


def test_app_host_wires_explicit_calibration_view_adapter() -> None:
    root = Path(__file__).resolve().parents[1]
    host_source = (root / "application" / "app_host.py").read_text(encoding="utf-8-sig")
    comp_source = (root / "application" / "composition.py").read_text(encoding="utf-8-sig")

    # Import and wiring now live in composition layer
    assert "from application.adapters.calibration_view import AppCalibrationViewAdapter" in comp_source
    assert "view=AppCalibrationViewAdapter(host)" in comp_source
    assert "host=self" not in comp_source[comp_source.index("calibration_controller = CalibrationController("):]

    # AppHost delegates to build_app_composition
    assert "build_app_composition(self)" in host_source


def test_app_host_does_not_wire_legacy_calibration_service_into_normal_runtime() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "application" / "app_host.py").read_text(encoding="utf-8-sig")

    assert "from services.calibration_service import CalibrationService" not in source
    assert "CalibrationService()" not in source
    assert "self.calibration_service" not in source
    assert "service=self.calibration_service" not in source


def test_od_gauge_sample_hook_does_not_instantiate_legacy_calibration_service() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "application" / "host" / "calibration" / "od.py").read_text(encoding="utf-8-sig")

    assert "from services.calibration_service import CalibrationService" not in source
    assert "CalibrationService()" not in source
