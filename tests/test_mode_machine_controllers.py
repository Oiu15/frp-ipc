from __future__ import annotations

from typing import cast

from services.calibration_controller import CalibrationController
from services.calibration_service import CalibrationService
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


class _FakeCalibrationService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name: str):
        def _recorder(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return _recorder


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

    def start_capture(self, *args, **kwargs) -> None:
        self.calls.append(("start_capture", args, kwargs))

    def compute_candidate(self, *args, **kwargs):
        self.calls.append(("compute_candidate", args, kwargs))
        return {"ok": True, "b_mm": 1.25, "n": 8}


class _FakeIdService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def start_capture(self, *args, **kwargs) -> None:
        self.calls.append(("start_capture", args, kwargs))

    def compute_candidate(self, *args, **kwargs):
        self.calls.append(("compute_candidate", args, kwargs))
        return {"ok": True, "delta_c_mm": 0.12}


class _FakeIdSingleService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def start_capture(self, *args, **kwargs) -> None:
        self.calls.append(("start_capture", args, kwargs))

    def compute_and_apply(self, *args, **kwargs):
        self.calls.append(("compute_and_apply", args, kwargs))
        return {"ok": True, "mean_l2_mm": 10.0, "b_mm": 0.5, "cov_pct": 98.0}


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
        service = _FakeCalibrationService()
        host = object()
        controller = CalibrationController(
            host=host,
            service=cast(CalibrationService, service),
            mode_machine=cast(ModeMachine, machine),
        )

        controller.compute_id_calibration()

        assert machine.entered == ["calibration"]
        assert machine.sync_calls == 1
        assert machine.runtime_state.mode_kind == "calibration"
        assert len(service.calls) == 1
        name, args, kwargs = service.calls[0]
        assert name == "compute_id_candidate"
        assert args == (host,)
        assert kwargs == {}

    def test_legacy_od_entrypoint_uses_injected_port_service(self) -> None:
        machine = _FakeModeMachine()
        legacy = _FakeCalibrationService()
        od_service = _FakeOdService()
        host = _FakeCalibrationHost()
        controller = CalibrationController(
            host=host,
            service=cast(CalibrationService, legacy),
            mode_machine=cast(ModeMachine, machine),
            od_service=cast(OdCalibrationService, od_service),
        )

        controller.start_od_b_capture()
        controller.compute_od_b()

        assert legacy.calls == []
        assert od_service.calls[0][0] == "start_capture"
        assert od_service.calls[0][2]["sampling_hz"] == 12.0
        assert od_service.calls[1] == ("compute_candidate", (181.0, 2.5), {})
        assert host.odcal_B_candidate_var.get() == "1.25000"

    def test_legacy_id_entrypoint_uses_injected_port_service(self) -> None:
        machine = _FakeModeMachine()
        legacy = _FakeCalibrationService()
        id_service = _FakeIdService()
        host = _FakeCalibrationHost()
        controller = CalibrationController(
            host=host,
            service=cast(CalibrationService, legacy),
            mode_machine=cast(ModeMachine, machine),
            id_service=cast(IdCalibrationService, id_service),
        )

        controller.start_id_capture()
        controller.compute_id_calibration()

        assert legacy.calls == []
        assert id_service.calls[0][0] == "start_capture"
        assert id_service.calls[0][2]["sampling_hz"] == 22.0
        assert id_service.calls[1] == ("compute_candidate", (151.0,), {})
        assert host.idcal_delta_candidate_var.get() == "0.1200"

    def test_legacy_id_single_entrypoint_uses_injected_port_service(self) -> None:
        machine = _FakeModeMachine()
        legacy = _FakeCalibrationService()
        id_single_service = _FakeIdSingleService()
        host = _FakeCalibrationHost()
        controller = CalibrationController(
            host=host,
            service=cast(CalibrationService, legacy),
            mode_machine=cast(ModeMachine, machine),
            id_single_service=cast(IdSingleCalibrationService, id_single_service),
        )

        controller.start_id_single_capture()
        controller.compute_and_write_id_single_calibration()

        assert legacy.calls == []
        assert id_single_service.calls[0][0] == "start_capture"
        assert id_single_service.calls[0][2]["sampling_hz"] == 22.0
        assert id_single_service.calls[1] == ("compute_and_apply", (152.0,), {})
        assert host.id_single_cal_B_var.get() == "0.50000"
