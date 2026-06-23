from __future__ import annotations

"""Application-level service / controller composition (Phase 2).

This module owns the "wiring" decisions — which concrete services to create,
how to thread their dependencies together, and where to attach them on the
AppHost instance.  The composition function is called once from
``AppHost.__init__`` and must preserve every legacy attribute so existing
code is undisturbed.
"""

from dataclasses import dataclass
from typing import Any

from application.adapters.calibration_view import AppCalibrationViewAdapter
from application.adapters.device_gateway import AppDeviceGateway
from services.calibration_controller import CalibrationController
from services.id_calibration import IdCalibrationService
from services.id_single_calibration import IdSingleCalibrationService
from services.measurement_service import MeasurementController
from services.od_calibration import OdCalibrationService
from services.tooling_calibration import ToolingCalibrationService
from services.results_service import ResultsService
from services.run_export_coordinator import RunExportCoordinator
from services.length_service import LengthService
from services.teach_service import TeachService
from modes.calibration_mode import CalibrationMode
from modes.mode_machine import ModeMachine
from modes.production_mode import ProductionMode
from modes.validation_mode import ValidationMode


@dataclass(slots=True)
class AppComposition:
    """Holder for services and controllers assembled during AppHost startup.

    Every field here is also attached to the AppHost instance as a
    backward-compatible attribute by ``build_app_composition(host)``.
    """

    results_service: ResultsService
    run_export_coordinator: RunExportCoordinator
    calibration_gateway: AppDeviceGateway
    od_calibration_svc: OdCalibrationService
    id_calibration_svc: IdCalibrationService
    id_single_calibration_svc: IdSingleCalibrationService
    calibration_mode: CalibrationMode
    validation_mode: ValidationMode
    production_mode: ProductionMode
    mode_machine: ModeMachine
    calibration_controller: CalibrationController
    measurement_controller: MeasurementController
    teach_service: TeachService
    length_service: LengthService


def build_app_composition(host: Any) -> AppComposition:
    """Create all application-layer services and controllers.

    This function is the single point of assembly.  Each block:
    1. Creates a service/controller.
    2. Attaches it to *host* (legacy attribute compatibility).
    3. Records it on the returned ``AppComposition``.

    The caller (AppHost.__init__) stores the result as
    ``self.composition = ...``.
    """
    results_service = ResultsService()
    host.results_service = results_service

    run_export_coordinator = RunExportCoordinator(
        repository=host._make_run_repository,
        results_service=results_service,
        recipe_provider=host.get_recipe_copy,
        calibration_provider=host.get_calibration_snapshot,
        coverage_provider=lambda: dict(host._section_cov_info or {}),
    )
    host._run_export_coordinator = run_export_coordinator

    calibration_gateway = AppDeviceGateway(host)
    host.calibration_gateway = calibration_gateway

    od_calibration_svc = OdCalibrationService(
        rotation=calibration_gateway,
        sensors=calibration_gateway,
        scheduler=calibration_gateway,
        state_sink=calibration_gateway,
        poll_profile=calibration_gateway,
        repository=host.calibration_repository,
    )
    host.od_calibration_svc = od_calibration_svc

    id_calibration_svc = IdCalibrationService(
        rotation=calibration_gateway,
        sensors=calibration_gateway,
        scheduler=calibration_gateway,
        state_sink=calibration_gateway,
        poll_profile=calibration_gateway,
        repository=host.calibration_repository,
    )
    host.id_calibration_svc = id_calibration_svc

    id_single_calibration_svc = IdSingleCalibrationService(
        rotation=calibration_gateway,
        sensors=calibration_gateway,
        scheduler=calibration_gateway,
        state_sink=calibration_gateway,
        poll_profile=calibration_gateway,
        repository=host.calibration_repository,
    )
    host.id_single_calibration_svc = id_single_calibration_svc

    tooling_calibration_svc = ToolingCalibrationService(
        rotation=calibration_gateway,
        sensors=calibration_gateway,
        scheduler=calibration_gateway,
        state_sink=calibration_gateway,
        poll_profile=calibration_gateway,
        repository=host.calibration_repository,
    )
    host.tooling_calibration_svc = tooling_calibration_svc

    calibration_mode = CalibrationMode()
    host.calibration_mode = calibration_mode

    validation_mode = ValidationMode(
        stop_impl=host.stop_validation_run,
        runner_getter=lambda: host._validation_thread,
    )
    host.validation_mode = validation_mode

    production_mode = ProductionMode(
        start_impl=host._start_measurement_impl,
        stop_impl=host._stop_measurement_impl,
        runner_getter=lambda: host._auto_thread,
        already_running_handler=lambda: host.show_warning(
            "Measurement", "Measurement is already running"
        ),
    )
    host.production_mode = production_mode

    mode_machine = ModeMachine(
        production_mode=production_mode,
        calibration_mode=calibration_mode,
        validation_mode=validation_mode,
        runtime_state=host.runtime_state,
    )
    host.mode_machine = mode_machine

    calibration_controller = CalibrationController(
        mode_machine=mode_machine,
        view=AppCalibrationViewAdapter(host),
        od_service=od_calibration_svc,
        id_service=id_calibration_svc,
        id_single_service=id_single_calibration_svc,
        tooling_service=tooling_calibration_svc,
    )
    host.calibration_controller = calibration_controller

    measurement_controller = MeasurementController(
        mode_machine=mode_machine,
    )
    host.measurement_controller = measurement_controller

    teach_service = TeachService(motion=host, operator=host, recipes=host)
    host.teach_service = teach_service

    length_service = LengthService()
    host.length_service = length_service

    return AppComposition(
        results_service=results_service,
        run_export_coordinator=run_export_coordinator,
        calibration_gateway=calibration_gateway,
        od_calibration_svc=od_calibration_svc,
        id_calibration_svc=id_calibration_svc,
        id_single_calibration_svc=id_single_calibration_svc,
        calibration_mode=calibration_mode,
        validation_mode=validation_mode,
        production_mode=production_mode,
        mode_machine=mode_machine,
        calibration_controller=calibration_controller,
        measurement_controller=measurement_controller,
        teach_service=teach_service,
        length_service=length_service,
    )


__all__ = ["AppComposition", "build_app_composition"]
