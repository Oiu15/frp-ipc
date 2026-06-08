import json
import shutil
import time
from pathlib import Path
from unittest.mock import patch

from machine.validation_gateway import ValidationActionCancelled
from domain.state import (
    CalibrationSnapshot,
    FixedSectionRepeatabilitySession,
    RuntimeState,
    VALIDATION_MOVE_CHANNELS,
    ValidationSession,
)
from core.models import AxisCal, MeasureRow, Recipe
from repositories.run_repository import RunRepository
from repositories.validation_repository import ValidationRepository
from tests.fakes import StrictDeviceGateway
from frp_workflow.validation_workflow import (
    FixedSectionRepeatabilityRequest,
    PhaseEvent,
    ValidationPhase,
    ValidationWorkflow,
    ValidationWorkflowEventType,
)
import pytest


def _phase_events(workflow: ValidationWorkflow) -> list[PhaseEvent]:
    return [event for event in workflow.events if isinstance(event, PhaseEvent)]


class RecordingValidationActionGateway(StrictDeviceGateway):
    def __init__(self) -> None:
        self.actions: list[object] = []
        self.angle_values: list[float] = [0.0, 3.0]
        self._angle_read_index = 0
        self.axis_positions: dict[int, float] = {0: 100.0, 1: 100.0, 2: 100.0, 4: 100.0}
        self.axis_cal = AxisCal()

    def stop_rotation(self) -> None:
        self.actions.append('stop_rotation')

    def clamp_release(self) -> None:
        self.actions.append('clamp_release')

    def clamp_close(self) -> None:
        self.actions.append('clamp_close')

    def wait_cancelable(self, duration_s: float, **_kwargs) -> None:
        self.actions.append(('wait_cancelable', float(duration_s)))

    def velmove(self, axis: int, velocity: float, **_kwargs) -> None:
        self.actions.append(('velmove', int(axis), float(velocity)))

    def read_axis_angle_deg_sync(self, axis: int = 3, timeout_s: float = 0.35) -> float:
        index = min(self._angle_read_index, len(self.angle_values) - 1)
        self._angle_read_index += 1
        return float(self.angle_values[index])

    def read_axis_position_mm(self, axis: int) -> float:
        value = float(self.axis_positions.get(int(axis), 0.0))
        self.actions.append(('read_axis_position_mm', int(axis), value))
        return value

    def get_axis_cal(self) -> AxisCal:
        return self.axis_cal

    def get_soft_limits_abs(self, _axes):
        return {}

    def move_axis_absolute(self, axis: int, target_pos_mm: float, *, context: str = 'ValidationMoveA') -> float:
        target = float(target_pos_mm)
        self.axis_positions[int(axis)] = target
        self.actions.append(('move_axis_absolute', int(axis), target, str(context)))
        return target

    def move_axes_absolute(self, targets_abs, *, context: str = 'ValidationMoveA'):
        resolved = {int(axis): float(target) for axis, target in dict(targets_abs).items()}
        for axis, target in resolved.items():
            self.axis_positions[int(axis)] = float(target)
        self.actions.append(('move_axes_absolute', dict(resolved), str(context)))
        return resolved

    def wait_axis_in_position(self, axis: int, target_pos_mm: float, **kwargs) -> float:
        target = float(target_pos_mm)
        self.axis_positions[int(axis)] = target
        self.actions.append(
            (
                'wait_axis_in_position',
                int(axis),
                target,
                float(kwargs.get('tolerance_mm', 0.0)),
                float(kwargs.get('timeout_s', 0.0)),
            )
        )
        return target

    def wait_axes_in_position(self, targets_abs, **kwargs):
        actuals = {}
        for axis, target in dict(targets_abs).items():
            actuals[int(axis)] = self.wait_axis_in_position(int(axis), float(target), **kwargs)
        return actuals


class CancelDuringWaitGateway(RecordingValidationActionGateway):
    def wait_cancelable(self, duration_s: float, **_kwargs) -> None:
        self.actions.append(('wait_cancelable', float(duration_s)))
        raise ValidationActionCancelled('validation stop requested')


class MoveInPositionTimeoutGateway(RecordingValidationActionGateway):
    def wait_axis_in_position(self, axis: int, target_pos_mm: float, **kwargs) -> float:
        target = float(target_pos_mm)
        self.actions.append(
            (
                'wait_axis_in_position',
                int(axis),
                target,
                float(kwargs.get('tolerance_mm', 0.0)),
                float(kwargs.get('timeout_s', 0.0)),
            )
        )
        raise TimeoutError('AX0 in-position timeout')


def _make_valid_section_result(od_avg: float = 100.0) -> MeasureRow:
    return MeasureRow(
        idx=1,
        x_ui=0.0,
        x_abs=0.0,
        od_avg=float(od_avg),
        od_dev=0.0,
        od_runout=0.0,
        od_round=0.0,
        id_avg=80.0,
        id_dev=0.0,
        id_runout=0.0,
        id_round=0.0,
        concentricity=0.0,
    )


def _make_valid_raw_points(od_mm: float = 100.0) -> list[dict]:
    return [
        {
            'phase': 'SYNC',
            'ts': float(i),
            'theta_deg': float(i * 10),
            'bin': i,
            'od_mm': float(od_mm),
        }
        for i in range(6)
    ]


def _make_valid_windows(raw_points: list[dict]) -> list[dict]:
    return [
        {
            'window_index': 1,
            'window_role': 'SYNC',
            'point_count': len(raw_points),
            'theta_span_deg': 50.0,
        }
    ]


class TestValidationWorkflowSmoke:
    def test_fixed_section_reclamp_request_and_state_fields(self) -> None:
        default_request = FixedSectionRepeatabilityRequest()
        assert not default_request.reclamp_enabled
        assert not default_request.rotation_stop_before_measure
        assert default_request.release_settle_s == 0.0
        assert default_request.clamp_settle_s == 0.0
        assert default_request.position_settle_s == 0.0
        assert default_request.sample_delay_s == 0.0
        assert default_request.validation_ax3_speed_dps == 60.0
        assert not default_request.move_enabled
        assert default_request.move_channel == 'od_channel'
        assert default_request.move_away_delta_mm == 0.0
        assert default_request.move_scenario == 'distance_round_trip'
        assert default_request.move_from_section_index == 1
        assert default_request.move_target_section_index == 1
        assert default_request.move_return_section_index == 1
        assert 'AX2' not in VALIDATION_MOVE_CHANNELS

        request = FixedSectionRepeatabilityRequest(
            reclamp_enabled=True,
            rotation_stop_before_measure=True,
            release_settle_s=0.25,
            clamp_settle_s=0.5,
            position_settle_s=0.75,
            sample_delay_s=0.125,
            validation_ax3_speed_dps=45.0,
            move_enabled=True,
            move_channel='id_channel',
            move_away_delta_mm=12.5,
            move_scenario='switch_and_return',
            move_from_section_index=1,
            move_target_section_index=2,
            move_return_section_index=1,
        )
        assert request.reclamp_enabled
        assert request.rotation_stop_before_measure
        assert request.release_settle_s == 0.25
        assert request.clamp_settle_s == 0.5
        assert request.position_settle_s == 0.75
        assert request.sample_delay_s == 0.125
        assert request.validation_ax3_speed_dps == 45.0
        assert request.move_enabled
        assert request.move_channel == 'id_channel'
        assert request.move_away_delta_mm == 12.5
        assert request.move_scenario == 'switch_and_return'
        assert request.move_from_section_index == 1
        assert request.move_target_section_index == 2
        assert request.move_return_section_index == 1

        session = FixedSectionRepeatabilitySession(
            reclamp_enabled=True,
            rotation_stop_before_measure=True,
            release_settle_s=0.25,
            clamp_settle_s=0.5,
            position_settle_s=0.75,
            sample_delay_s=0.125,
            validation_ax3_speed_dps=45.0,
            move_enabled=True,
            move_channel='od_id_sync',
            move_away_delta_mm=12.5,
            move_scenario='switch_and_measure_target',
            move_from_section_index=1,
            move_target_section_index=2,
            move_return_section_index=2,
        )
        assert session.reclamp_enabled
        assert session.rotation_stop_before_measure
        assert session.release_settle_s == 0.25
        assert session.clamp_settle_s == 0.5
        assert session.position_settle_s == 0.75
        assert session.sample_delay_s == 0.125
        assert session.validation_ax3_speed_dps == 45.0
        assert session.move_enabled
        assert session.move_channel == 'od_id_sync'
        assert session.move_away_delta_mm == 12.5
        assert session.move_scenario == 'switch_and_measure_target'
        assert session.move_from_section_index == 1
        assert session.move_target_section_index == 2
        assert session.move_return_section_index == 2

    def test_fixed_section_before_capture_runs_configured_reclamp_actions(self, request) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        tmp_root = repo_root / '.compile_check' / 'validation_before_capture_reclamp'
        tmp_root.mkdir(parents=True, exist_ok=True)
        case_root = tmp_root / f'case_{int(time.time() * 1000)}'
        request.addfinalizer(lambda: shutil.rmtree(case_root, True))
        app_root = case_root / 'FRP_IPC'
        app_root.mkdir(parents=True, exist_ok=True)

        gateway = RecordingValidationActionGateway()
        session = ValidationSession()
        workflow = ValidationWorkflow(
            recipe=Recipe(name='validation-before-capture', rot_vel_velmove=180.0),
            calibration=CalibrationSnapshot(),
            runtime_state=RuntimeState.from_validation_session(session),
            gateway=gateway,
            run_repository=RunRepository(app_root_dir=app_root),
            validation_session=session,
        )
        request = FixedSectionRepeatabilityRequest(
            section_name='S1',
            metric_name='od_avg',
            repeat_count=1,
            reclamp_enabled=True,
            rotation_stop_before_measure=True,
            release_settle_s=0.25,
            clamp_settle_s=0.5,
            position_settle_s=0.75,
            sample_delay_s=0.125,
            validation_ax3_speed_dps=45.0,
        )
        section_result = _make_valid_section_result()
        raw_points = _make_valid_raw_points()
        windows = _make_valid_windows(raw_points)

        def _capture_side_effect(**_kwargs):
            gateway.actions.append('capture')
            return section_result, raw_points, windows, {'cov': 1.0}

        with patch(
            'frp_workflow.validation_workflow.measure_current_position_section_capture',
            side_effect=_capture_side_effect,
        ) as capture_mock:
            rows, summary = workflow.run_fixed_section_repeatability(request)

        capture_mock.assert_called_once()
        assert len(rows) == 1
        assert summary['count'] == 1
        assert gateway.actions == [
                'stop_rotation',
                'clamp_release',
                ('wait_cancelable', 0.25),
                'clamp_close',
                ('wait_cancelable', 0.5),
                ('velmove', 3, 45.0),
                ('wait_cancelable', 0.05),
                ('wait_cancelable', 0.75),
                ('wait_cancelable', 0.125),
                'capture',
            ]
        phase_events = _phase_events(workflow)
        assert [event.phase for event in phase_events] == [
                ValidationPhase.PREPARE.value,
                ValidationPhase.BEFORE_CAPTURE.value,
                ValidationPhase.STOP_ROTATION.value,
                ValidationPhase.UNCLAMP.value,
                ValidationPhase.WAIT_UNCLAMP_SETTLE.value,
                ValidationPhase.CLAMP.value,
                ValidationPhase.WAIT_CLAMP_SETTLE.value,
                ValidationPhase.RESTORE_ROTATION_READY.value,
                ValidationPhase.WAIT_POSITION_SETTLE.value,
                ValidationPhase.WAIT_SAMPLE_DELAY.value,
                ValidationPhase.CAPTURE.value,
                ValidationPhase.FIT_CALC.value,
                ValidationPhase.SAVE_RESULT.value,
            ]

    def test_fixed_section_before_capture_runs_section_relocation_actions(self, request) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        tmp_root = repo_root / '.compile_check' / 'validation_before_capture_move'
        tmp_root.mkdir(parents=True, exist_ok=True)
        case_root = tmp_root / f'case_{int(time.time() * 1000)}'
        request.addfinalizer(lambda: shutil.rmtree(case_root, True))
        app_root = case_root / 'FRP_IPC'
        app_root.mkdir(parents=True, exist_ok=True)

        gateway = RecordingValidationActionGateway()
        session = ValidationSession()
        workflow = ValidationWorkflow(
            recipe=Recipe(name='validation-before-capture-move'),
            calibration=CalibrationSnapshot(),
            runtime_state=RuntimeState.from_validation_session(session),
            gateway=gateway,
            run_repository=RunRepository(app_root_dir=app_root),
            validation_session=session,
        )
        request = FixedSectionRepeatabilityRequest(
            section_name='S1',
            metric_name='od_avg',
            repeat_count=1,
            move_enabled=True,
            move_channel='od_channel',
            move_away_delta_mm=12.5,
        )
        section_result = _make_valid_section_result()
        raw_points = _make_valid_raw_points()
        windows = _make_valid_windows(raw_points)

        def _capture_side_effect(**_kwargs):
            gateway.actions.append('capture')
            return section_result, raw_points, windows, {'cov': 1.0}

        phase_updates = []
        with patch(
            'frp_workflow.validation_workflow.measure_current_position_section_capture',
            side_effect=_capture_side_effect,
        ) as capture_mock:
            rows, summary = workflow.run_fixed_section_repeatability(
                request,
                phase_callback=lambda event: phase_updates.append(event),
            )

        capture_mock.assert_called_once()
        assert len(rows) == 1
        assert summary['count'] == 1
        assert gateway.actions == [
                ('read_axis_position_mm', 0, 100.0),
                ('move_axes_absolute', {0: 87.5}, 'VALIDATION_MOVE_AWAY'),
                ('wait_axis_in_position', 0, 87.5, 0.1, 10.0),
                ('move_axes_absolute', {0: 100.0}, 'VALIDATION_MOVE_BACK_TO_TARGET'),
                ('wait_axis_in_position', 0, 100.0, 0.1, 10.0),
                'capture',
            ]
        phase_events = _phase_events(workflow)
        assert [event.phase for event in phase_events] == [
                ValidationPhase.PREPARE.value,
                ValidationPhase.BEFORE_CAPTURE.value,
                ValidationPhase.MOVE_AWAY.value,
                ValidationPhase.MOVE_BACK_TO_TARGET.value,
                ValidationPhase.WAIT_POSITION_SETTLE.value,
                ValidationPhase.WAIT_SAMPLE_DELAY.value,
                ValidationPhase.CAPTURE.value,
                ValidationPhase.FIT_CALC.value,
                ValidationPhase.SAVE_RESULT.value,
            ]
        move_updates = [
            (event.phase, dict(event.payload))
            for event in phase_updates
            if event.phase in {
                ValidationPhase.MOVE_AWAY.value,
                ValidationPhase.MOVE_BACK_TO_TARGET.value,
            }
        ]
        assert [
                (
                    phase,
                    payload.get('move_channel'),
                    payload.get('target_positions_mm'),
                    payload.get('actual_positions_mm'),
                )
                for phase, payload in move_updates
            ] == [
                (ValidationPhase.MOVE_AWAY.value, 'od_channel', {'AX0': 87.5}, {'AX0': 100.0}),
                (ValidationPhase.MOVE_AWAY.value, 'od_channel', {'AX0': 87.5}, {'AX0': 87.5}),
                (ValidationPhase.MOVE_BACK_TO_TARGET.value, 'od_channel', {'AX0': 100.0}, {'AX0': 87.5}),
                (ValidationPhase.MOVE_BACK_TO_TARGET.value, 'od_channel', {'AX0': 100.0}, {'AX0': 100.0}),
            ]

    def test_fixed_section_before_capture_uses_id_channel_target_planning(self, request) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        tmp_root = repo_root / '.compile_check' / 'validation_before_capture_id_move'
        tmp_root.mkdir(parents=True, exist_ok=True)
        case_root = tmp_root / f'case_{int(time.time() * 1000)}'
        request.addfinalizer(lambda: shutil.rmtree(case_root, True))
        app_root = case_root / 'FRP_IPC'
        app_root.mkdir(parents=True, exist_ok=True)

        gateway = RecordingValidationActionGateway()
        gateway.axis_positions.update({1: 50.0, 4: 50.0, 2: 0.0})
        session = ValidationSession()
        workflow = ValidationWorkflow(
            recipe=Recipe(name='validation-before-capture-id-move'),
            calibration=CalibrationSnapshot(),
            runtime_state=RuntimeState.from_validation_session(session),
            gateway=gateway,
            run_repository=RunRepository(app_root_dir=app_root),
            validation_session=session,
        )
        request = FixedSectionRepeatabilityRequest(
            section_name='S1',
            metric_name='id_avg',
            repeat_count=1,
            move_enabled=True,
            move_channel='id_channel',
            move_away_delta_mm=20.0,
        )
        section_result = _make_valid_section_result()
        raw_points = _make_valid_raw_points()
        windows = _make_valid_windows(raw_points)

        def _capture_side_effect(**_kwargs):
            gateway.actions.append('capture')
            return section_result, raw_points, windows, {'cov': 1.0}

        with patch(
            'frp_workflow.validation_workflow.measure_current_position_section_capture',
            side_effect=_capture_side_effect,
        ):
            rows, summary = workflow.run_fixed_section_repeatability(request)

        assert len(rows) == 1
        assert summary['count'] == 1
        assert gateway.actions == [
                ('read_axis_position_mm', 1, 50.0),
                ('read_axis_position_mm', 4, 50.0),
                ('move_axes_absolute', {1: 40.0, 4: 40.0}, 'VALIDATION_MOVE_AWAY'),
                ('wait_axis_in_position', 1, 40.0, 0.1, 10.0),
                ('wait_axis_in_position', 4, 40.0, 0.1, 10.0),
                ('move_axes_absolute', {1: 50.0, 4: 50.0}, 'VALIDATION_MOVE_BACK_TO_TARGET'),
                ('wait_axis_in_position', 1, 50.0, 0.1, 10.0),
                ('wait_axis_in_position', 4, 50.0, 0.1, 10.0),
                'capture',
            ]

    def test_fixed_section_formal_section_switch_records_planned_and_actual_targets(self, request) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        tmp_root = repo_root / '.compile_check' / 'validation_before_capture_section_switch'
        tmp_root.mkdir(parents=True, exist_ok=True)
        case_root = tmp_root / f'case_{int(time.time() * 1000)}'
        request.addfinalizer(lambda: shutil.rmtree(case_root, True))
        app_root = case_root / 'FRP_IPC'
        app_root.mkdir(parents=True, exist_ok=True)

        gateway = RecordingValidationActionGateway()
        session = ValidationSession()
        workflow = ValidationWorkflow(
            recipe=Recipe(
                name='validation-before-capture-section-switch',
                section_count=3,
                section_pos_z=[0.0, 20.0, 40.0],
            ),
            calibration=CalibrationSnapshot(),
            runtime_state=RuntimeState.from_validation_session(session),
            gateway=gateway,
            run_repository=RunRepository(app_root_dir=app_root),
            validation_session=session,
        )
        request = FixedSectionRepeatabilityRequest(
            section_name='S2',
            metric_name='od_avg',
            repeat_count=1,
            move_enabled=True,
            move_channel='od_id_sync',
            move_scenario='switch_and_return',
            move_from_section_index=1,
            move_target_section_index=2,
            move_return_section_index=3,
        )
        section_result = MeasureRow(
            idx=3,
            x_ui=40.0,
            x_abs=0.0,
            od_avg=100.0,
            od_dev=0.0,
            od_runout=0.0,
            od_round=0.0,
            id_avg=80.0,
            id_dev=0.0,
            id_runout=0.0,
            id_round=0.0,
            concentricity=0.0,
        )
        raw_points = _make_valid_raw_points()
        windows = _make_valid_windows(raw_points)

        def _capture_side_effect(**_kwargs):
            gateway.actions.append('capture')
            return section_result, raw_points, windows, {'cov': 1.0}

        phase_updates = []
        with patch(
            'frp_workflow.validation_workflow.measure_current_position_section_capture',
            side_effect=_capture_side_effect,
        ):
            rows, summary = workflow.run_fixed_section_repeatability(
                request,
                phase_callback=lambda event: phase_updates.append(event),
            )

        assert len(rows) == 1
        assert summary['count'] == 1
        phase_events = _phase_events(workflow)
        assert [event.phase for event in phase_events] == [
                ValidationPhase.PREPARE.value,
                ValidationPhase.BEFORE_CAPTURE.value,
                ValidationPhase.MOVE_TO_FROM_SECTION.value,
                ValidationPhase.MOVE_TO_TARGET_SECTION.value,
                ValidationPhase.MOVE_TO_RETURN_SECTION.value,
                ValidationPhase.WAIT_POSITION_SETTLE.value,
                ValidationPhase.WAIT_SAMPLE_DELAY.value,
                ValidationPhase.CAPTURE.value,
                ValidationPhase.FIT_CALC.value,
                ValidationPhase.SAVE_RESULT.value,
            ]
        reached_target_payload = [
            dict(event.payload)
            for event in phase_updates
            if event.phase == ValidationPhase.MOVE_TO_TARGET_SECTION.value
        ][-1]
        assert reached_target_payload['from_section_index'] == 1
        assert reached_target_payload['target_section_index'] == 2
        assert reached_target_payload['return_section_index'] == 3
        assert reached_target_payload['section_index'] == 2
        assert reached_target_payload['z_pos_mm'] == 20.0
        assert reached_target_payload['measure_section_index'] == 3
        assert reached_target_payload['measure_section_name'] == '3: 40.000'
        assert reached_target_payload['measured_z_pos_mm'] == 40.0
        assert reached_target_payload['planned_targets_mm'] == {'AX0': -20.0, 'AX1': -10.0, 'AX4': -10.0}
        assert reached_target_payload['actual_positions_after_wait_mm'] == {'AX0': -20.0, 'AX1': -10.0, 'AX4': -10.0}
        assert rows[0].section_name == '3: 40.000'
        assert rows[0].measure_section_index == 3
        assert rows[0].measure_section_name == '3: 40.000'
        assert rows[0].measured_z_pos_mm == 40.0
        assert summary['measure_section_index'] == 3
        assert summary['measure_section_name'] == '3: 40.000'
        assert summary['measured_z_pos_mm'] == 40.0
        assert workflow.fixed_section_repeat_captures[0].raw_points[0]['section_idx'] == 3

    def test_fixed_section_switch_and_measure_target_stays_on_target_section(self, request) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        tmp_root = repo_root / '.compile_check' / 'validation_before_capture_switch_target'
        tmp_root.mkdir(parents=True, exist_ok=True)
        case_root = tmp_root / f'case_{int(time.time() * 1000)}'
        request.addfinalizer(lambda: shutil.rmtree(case_root, True))
        app_root = case_root / 'FRP_IPC'
        app_root.mkdir(parents=True, exist_ok=True)

        gateway = RecordingValidationActionGateway()
        session = ValidationSession()
        workflow = ValidationWorkflow(
            recipe=Recipe(
                name='validation-before-capture-switch-target',
                section_count=3,
                section_pos_z=[0.0, 20.0, 40.0],
            ),
            calibration=CalibrationSnapshot(),
            runtime_state=RuntimeState.from_validation_session(session),
            gateway=gateway,
            run_repository=RunRepository(app_root_dir=app_root),
            validation_session=session,
        )
        request = FixedSectionRepeatabilityRequest(
            section_name='S2',
            metric_name='od_avg',
            repeat_count=1,
            move_enabled=True,
            move_channel='od_id_sync',
            move_scenario='switch_and_measure_target',
            move_from_section_index=1,
            move_target_section_index=2,
            move_return_section_index=3,
        )
        section_result = MeasureRow(
            idx=2,
            x_ui=20.0,
            x_abs=0.0,
            od_avg=100.0,
            od_dev=0.0,
            od_runout=0.0,
            od_round=0.0,
            id_avg=80.0,
            id_dev=0.0,
            id_runout=0.0,
            id_round=0.0,
            concentricity=0.0,
        )

        with patch(
            'frp_workflow.validation_workflow.measure_current_position_section_capture',
            return_value=(section_result, _make_valid_raw_points(), _make_valid_windows(_make_valid_raw_points()), {'cov': 1.0}),
        ):
            rows, summary = workflow.run_fixed_section_repeatability(request)

        phase_events = _phase_events(workflow)
        assert ValidationPhase.MOVE_TO_RETURN_SECTION.value not in [event.phase for event in phase_events]
        assert rows[0].section_name == '2: 20.000'
        assert rows[0].measure_section_index == 2
        assert rows[0].measure_section_name == '2: 20.000'
        assert rows[0].measured_z_pos_mm == 20.0
        assert summary['measure_section_index'] == 2
        assert summary['measure_section_name'] == '2: 20.000'

    def test_fixed_section_relocation_wait_failure_blocks_capture(self, request) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        tmp_root = repo_root / '.compile_check' / 'validation_before_capture_move_timeout'
        tmp_root.mkdir(parents=True, exist_ok=True)
        case_root = tmp_root / f'case_{int(time.time() * 1000)}'
        request.addfinalizer(lambda: shutil.rmtree(case_root, True))
        app_root = case_root / 'FRP_IPC'
        app_root.mkdir(parents=True, exist_ok=True)

        gateway = MoveInPositionTimeoutGateway()
        session = ValidationSession()
        workflow = ValidationWorkflow(
            recipe=Recipe(name='validation-before-capture-move-timeout'),
            calibration=CalibrationSnapshot(),
            runtime_state=RuntimeState.from_validation_session(session),
            gateway=gateway,
            run_repository=RunRepository(app_root_dir=app_root),
            validation_session=session,
        )
        request = FixedSectionRepeatabilityRequest(
            section_name='S1',
            metric_name='od_avg',
            repeat_count=1,
            move_enabled=True,
            move_channel='od_channel',
            move_away_delta_mm=12.5,
        )

        with patch('frp_workflow.validation_workflow.measure_current_position_section_capture') as capture_mock:
            with pytest.raises(TimeoutError):
                workflow.run_fixed_section_repeatability(request)

        capture_mock.assert_not_called()
        phase_events = _phase_events(workflow)
        assert [event.phase for event in phase_events] == [
                ValidationPhase.PREPARE.value,
                ValidationPhase.BEFORE_CAPTURE.value,
                ValidationPhase.MOVE_AWAY.value,
            ]
        assert workflow.current_phase == ValidationPhase.MOVE_AWAY
        assert workflow.runtime_state.status == 'error'

    def test_fixed_section_cancel_during_before_capture_does_not_hang_or_capture(self, request) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        tmp_root = repo_root / '.compile_check' / 'validation_before_capture_cancel'
        tmp_root.mkdir(parents=True, exist_ok=True)
        case_root = tmp_root / f'case_{int(time.time() * 1000)}'
        request.addfinalizer(lambda: shutil.rmtree(case_root, True))
        app_root = case_root / 'FRP_IPC'
        app_root.mkdir(parents=True, exist_ok=True)

        gateway = CancelDuringWaitGateway()
        session = ValidationSession()
        workflow = ValidationWorkflow(
            recipe=Recipe(name='validation-before-capture-cancel'),
            calibration=CalibrationSnapshot(),
            runtime_state=RuntimeState.from_validation_session(session),
            gateway=gateway,
            run_repository=RunRepository(app_root_dir=app_root),
            validation_session=session,
        )
        request = FixedSectionRepeatabilityRequest(
            section_name='S1',
            metric_name='od_avg',
            repeat_count=1,
            reclamp_enabled=True,
            rotation_stop_before_measure=True,
            release_settle_s=60.0,
            clamp_settle_s=60.0,
            position_settle_s=60.0,
            sample_delay_s=60.0,
            validation_ax3_speed_dps=45.0,
        )

        t0 = time.monotonic()
        with patch('frp_workflow.validation_workflow.measure_current_position_section_capture') as capture_mock:
            with pytest.raises(ValidationActionCancelled):
                workflow.run_fixed_section_repeatability(request)
        elapsed_s = time.monotonic() - t0

        assert elapsed_s < 0.5
        capture_mock.assert_not_called()
        assert gateway.actions == [
                'stop_rotation',
                'clamp_release',
                ('wait_cancelable', 60.0),
            ]
        phase_events = _phase_events(workflow)
        assert [event.phase for event in phase_events] == [
                ValidationPhase.PREPARE.value,
                ValidationPhase.BEFORE_CAPTURE.value,
                ValidationPhase.STOP_ROTATION.value,
                ValidationPhase.UNCLAMP.value,
                ValidationPhase.WAIT_UNCLAMP_SETTLE.value,
            ]
        assert workflow.current_phase == ValidationPhase.WAIT_UNCLAMP_SETTLE
        assert workflow.runtime_state.status == 'error'
        result = workflow.result
        assert result is not None
        assert result is not None
        assert result.status == 'ERR'
        assert len(workflow.fixed_section_repeat_captures) == 0
        assert len(workflow.runtime_state.rows) == 0

    def test_fixed_section_blocks_capture_when_rotation_is_not_ready(self, request) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        tmp_root = repo_root / '.compile_check' / 'validation_rotation_not_ready'
        tmp_root.mkdir(parents=True, exist_ok=True)
        case_root = tmp_root / f'case_{int(time.time() * 1000)}'
        request.addfinalizer(lambda: shutil.rmtree(case_root, True))
        app_root = case_root / 'FRP_IPC'
        app_root.mkdir(parents=True, exist_ok=True)

        gateway = RecordingValidationActionGateway()
        gateway.angle_values = [0.0, 0.0, 0.0]
        session = ValidationSession()
        workflow = ValidationWorkflow(
            recipe=Recipe(name='validation-rotation-not-ready'),
            calibration=CalibrationSnapshot(),
            runtime_state=RuntimeState.from_validation_session(session),
            gateway=gateway,
            run_repository=RunRepository(app_root_dir=app_root),
            validation_session=session,
        )
        request = FixedSectionRepeatabilityRequest(
            section_name='S1',
            metric_name='od_avg',
            repeat_count=1,
            reclamp_enabled=True,
            rotation_stop_before_measure=True,
            release_settle_s=0.0,
            clamp_settle_s=0.0,
            validation_ax3_speed_dps=45.0,
        )

        with patch('frp_workflow.validation_workflow._ROTATION_READY_TIMEOUT_S', 0.01):
            with patch('frp_workflow.validation_workflow.measure_current_position_section_capture') as capture_mock:
                with pytest.raises(RuntimeError, match='AX3'):
                    workflow.run_fixed_section_repeatability(request)

        capture_mock.assert_not_called()
        phase_events = _phase_events(workflow)
        assert [event.phase for event in phase_events] == [
                ValidationPhase.PREPARE.value,
                ValidationPhase.BEFORE_CAPTURE.value,
                ValidationPhase.STOP_ROTATION.value,
                ValidationPhase.UNCLAMP.value,
                ValidationPhase.WAIT_UNCLAMP_SETTLE.value,
                ValidationPhase.CLAMP.value,
                ValidationPhase.WAIT_CLAMP_SETTLE.value,
                ValidationPhase.RESTORE_ROTATION_READY.value,
            ]
        assert workflow.current_phase == ValidationPhase.RESTORE_ROTATION_READY
        assert workflow.runtime_state.status == 'error'
        result = workflow.result
        assert result is not None
        assert result is not None
        assert 'AX3' in result.message

    def test_smoke_events_result_and_export(self, request) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        tmp_root = repo_root / '.compile_check' / 'validation_workflow_smoke'
        tmp_root.mkdir(parents=True, exist_ok=True)
        case_root = tmp_root / f'case_{int(time.time() * 1000)}'
        request.addfinalizer(lambda: shutil.rmtree(case_root, True))
        app_root = case_root / 'FRP_IPC'
        app_root.mkdir(parents=True, exist_ok=True)

        session = ValidationSession(
            standard_piece_id='STD-RING-001',
            validation_batch_id='VAL-20260408-A',
            repeat_measurement_count=3,
        )
        workflow = ValidationWorkflow(
            recipe=Recipe(name='validation-smoke'),
            calibration=CalibrationSnapshot(),
            runtime_state=RuntimeState.from_validation_session(session),
            gateway=StrictDeviceGateway(),
            run_repository=RunRepository(app_root_dir=app_root),
            validation_session=session,
        )

        identity = workflow.ensure_identity()
        workflow.record_state('PREP', 'prepare validation')
        workflow.record_progress(step='acquire_baseline', index=1, total=2, message='collecting')
        workflow.record_summary({'baseline_ok': True, 'delta_mm': 0.012}, source='baseline')
        workflow.record_state('DONE', 'completed')
        result = workflow.build_result(status='DONE', message='completed', finished_at_ts=time.time())

        export_repo = ValidationRepository(app_root_dir=app_root)
        export_ctx = workflow.build_export_context()
        run_dir = Path(export_repo.export_run(export_ctx))

        assert identity.serial == workflow.runtime_state.serial
        assert identity.serial == session.serial
        assert identity.run_id == session.run_id
        assert workflow.runtime_state.status == 'completed'
        assert result.status == 'DONE'
        assert result.identity == identity
        assert result.finished_at_ts is not None
        assert result.standard_piece_id == 'STD-RING-001'
        assert result.validation_batch_id == 'VAL-20260408-A'
        assert result.repeat_measurement_count == 3
        assert result.summary['baseline_ok'] == True
        assert session.summary_cache['baseline_ok'] == True
        assert session.summary_cache['delta_mm'] == 0.012
        assert [event.type for event in workflow.events] == [
                ValidationWorkflowEventType.STATE,
                ValidationWorkflowEventType.PROGRESS,
                ValidationWorkflowEventType.SUMMARY,
                ValidationWorkflowEventType.STATE,
            ]

        assert run_dir.parent.parent.name == 'validation_exports'
        assert (run_dir / 'validation_result.json').exists()
        assert (run_dir / 'validation_events.json').exists()
        assert (run_dir.parent / 'summary.csv').exists()
        assert not (app_root / 'exports').exists()

        payload = json.loads((run_dir / 'validation_result.json').read_text(encoding='utf-8'))
        assert payload['serial'] == identity.serial
        assert payload['standard_piece_id'] == 'STD-RING-001'
        assert payload['validation_batch_id'] == 'VAL-20260408-A'
        assert payload['repeat_measurement_count'] == 3

    def test_fixed_section_repeatability_smoke_runs_sampling_chain(self, request) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        tmp_root = repo_root / '.compile_check' / 'validation_fixed_section_smoke'
        tmp_root.mkdir(parents=True, exist_ok=True)
        case_root = tmp_root / f'case_{int(time.time() * 1000)}'
        request.addfinalizer(lambda: shutil.rmtree(case_root, True))
        app_root = case_root / 'FRP_IPC'
        app_root.mkdir(parents=True, exist_ok=True)

        session = ValidationSession()
        gateway = RecordingValidationActionGateway()
        workflow = ValidationWorkflow(
            recipe=Recipe(name='validation-fixed-section-smoke'),
            calibration=CalibrationSnapshot(),
            runtime_state=RuntimeState.from_validation_session(session),
            gateway=gateway,
            run_repository=RunRepository(app_root_dir=app_root),
            validation_session=session,
        )
        request = FixedSectionRepeatabilityRequest(
            section_name='S1',
            metric_name='od_avg',
            repeat_count=1,
            position_settle_s=0.2,
            sample_delay_s=0.1,
        )
        section_result = MeasureRow(
            idx=1,
            x_ui=12.0,
            x_abs=34.0,
            od_avg=123.456,
            od_dev=0.156,
            od_runout=0.01,
            od_round=0.02,
            id_avg=80.0,
            id_dev=0.0,
            id_runout=0.0,
            id_round=0.0,
            concentricity=0.0,
        )
        raw_points = [
            {
                'phase': 'SYNC',
                'ts': float(i),
                'theta_deg': float(i * 10),
                'bin': i,
                'od_mm': 123.456,
            }
            for i in range(6)
        ]
        windows = [
            {
                'window_index': 1,
                'window_role': 'SYNC',
                'point_start_index': 0,
                'point_end_index': 5,
                'point_count': len(raw_points),
                'ts_start': 0.0,
                'ts_end': 5.0,
                'theta_start_deg': 0.0,
                'theta_end_deg': 50.0,
                'theta_span_deg': 50.0,
                'filled_bins': 6,
                'total_bins': 6,
                'miss_bins': 0,
                'n_od': 6,
                'n_id': 0,
                'reason': 'COV',
                'revs': 0.2,
                'elapsed_s': 0.5,
                'max_gap_deg': 10.0,
            }
        ]
        fit_payload = {
            'od_center_x_mm': 0.12,
            'od_center_y_mm': -0.34,
            'od_radius_mm': 61.728,
            'od_diameter_fit_mm': 123.456,
            'id_center_x_mm': 0.02,
            'id_center_y_mm': -0.03,
            'id_radius_mm': 40.0,
            'id_diameter_fit_mm': 80.0,
            'od_ecc_mm': None,
            'id_ecc_mm': None,
            'concentricity_mm': 0.321,
        }

        with patch(
            'frp_workflow.validation_workflow.measure_current_position_section_capture',
            return_value=(section_result, raw_points, windows, {'cov': 1.0}, fit_payload),
        ) as capture_mock:
            rows, summary = workflow.run_fixed_section_repeatability(request)

        capture_mock.assert_called_once()
        capture_kwargs = capture_mock.call_args.kwargs
        assert capture_kwargs['gateway'] is workflow.gateway
        assert capture_kwargs['recipe'] is workflow.recipe
        assert capture_kwargs['calibration'] is workflow.calibration
        assert len(rows) == 1
        assert rows[0].repeat_index == 1
        assert rows[0].section_name == 'current: 12.000'
        assert rows[0].measure_section_index is None
        assert rows[0].measure_section_name == 'current: 12.000'
        assert rows[0].measured_z_pos_mm == 12.0
        assert rows[0].metric_name == 'od_avg'
        assert rows[0].measured_value_mm == 123.456
        assert rows[0].settle_s_used == 0.2
        assert rows[0].sample_delay_s_used == 0.1
        assert rows[0].capture_start_ts == 0.0
        assert rows[0].capture_end_ts == 5.0
        assert summary['count'] == 1
        assert summary['measure_section_index'] is None
        assert summary['measure_section_name'] == 'current: 12.000'
        assert summary['measured_z_pos_mm'] == 12.0
        assert summary['primary_metric']['od_avg']['count'] == 1
        assert summary['primary_metric']['od_avg']['mean'] == 123.456
        assert workflow.runtime_state.status == 'completed'
        result = workflow.result
        assert result is not None
        assert result is not None
        assert result.status == 'DONE'
        assert session.repeat_measurement_count == 1
        assert gateway.actions == [
                ('wait_cancelable', 0.2),
                ('wait_cancelable', 0.1),
            ]
        assert len(workflow.runtime_state.rows) == 1
        assert len(workflow.runtime_state.raw_points) == 6
        assert len(workflow.fixed_section_repeat_captures) == 1
        phase_events = _phase_events(workflow)
        assert [event.phase for event in phase_events] == [
                ValidationPhase.PREPARE.value,
                ValidationPhase.BEFORE_CAPTURE.value,
                ValidationPhase.WAIT_POSITION_SETTLE.value,
                ValidationPhase.WAIT_SAMPLE_DELAY.value,
                ValidationPhase.CAPTURE.value,
                ValidationPhase.FIT_CALC.value,
                ValidationPhase.SAVE_RESULT.value,
            ]
        capture = workflow.fixed_section_repeat_captures[0]
        assert capture.section_result == section_result
        assert capture.section_name == 'current: 12.000'
        assert capture.measure_section_index is None
        assert capture.measure_section_name == 'current: 12.000'
        assert capture.measured_z_pos_mm == 12.0
        fit_result = capture.fit_result
        assert fit_result is not None
        assert fit_result is not None
        assert fit_result.measure_section_name == 'current: 12.000'
        assert fit_result.measured_z_pos_mm == 12.0
        assert fit_result.od_center_x_mm == 0.12
        assert fit_result.od_center_y_mm == -0.34
        assert fit_result.od_radius_mm == 61.728
        assert fit_result.od_diameter_fit_mm == 123.456
        assert fit_result.id_center_x_mm == 0.02
        assert fit_result.id_center_y_mm == -0.03
        assert fit_result.id_radius_mm == 40.0
        assert fit_result.id_diameter_fit_mm == 80.0
        assert fit_result.concentricity_mm == 0.321
        assert capture.settle_s_used == 0.2
        assert capture.sample_delay_s_used == 0.1
        assert capture.capture_start_ts == 0.0
        assert capture.capture_end_ts == 5.0
        assert len(capture.raw_points) == 6
        assert capture.raw_points[0]['section_idx'] is None
        assert capture.raw_points[0]['measure_section_name'] == 'current: 12.000'
        assert capture.raw_points[0]['measured_z_pos_mm'] == 12.0
        assert len(capture.windows) == 1
        assert capture.windows[0].point_count == 6
        assert capture.coverage['cov'] == 1.0

        export_context = workflow.build_export_context()
        assert export_context.status == 'DONE'
        assert export_context.repeat_measurement_count == 1
        assert export_context.summary['count'] == 1
        assert export_context.summary['measure_section_name'] == 'current: 12.000'

    def test_fixed_section_wait_callback_reports_remaining_time(self, request) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        tmp_root = repo_root / '.compile_check' / 'validation_wait_callback'
        tmp_root.mkdir(parents=True, exist_ok=True)
        case_root = tmp_root / f'case_{int(time.time() * 1000)}'
        request.addfinalizer(lambda: shutil.rmtree(case_root, True))
        app_root = case_root / 'FRP_IPC'
        app_root.mkdir(parents=True, exist_ok=True)

        gateway = RecordingValidationActionGateway()
        session = ValidationSession()
        workflow = ValidationWorkflow(
            recipe=Recipe(name='validation-wait-callback'),
            calibration=CalibrationSnapshot(),
            runtime_state=RuntimeState.from_validation_session(session),
            gateway=gateway,
            run_repository=RunRepository(app_root_dir=app_root),
            validation_session=session,
        )
        request = FixedSectionRepeatabilityRequest(
            section_name='S1',
            metric_name='od_avg',
            repeat_count=1,
            position_settle_s=0.2,
            sample_delay_s=0.1,
        )
        wait_seen: list[tuple[str, int, int, float]] = []

        with patch(
            'frp_workflow.validation_workflow.measure_current_position_section_capture',
            return_value=(_make_valid_section_result(), _make_valid_raw_points(), _make_valid_windows(_make_valid_raw_points()), {'cov': 1.0}),
        ):
            workflow.run_fixed_section_repeatability(
                request,
                wait_callback=lambda phase, repeat_index, total, remaining_s: wait_seen.append(
                    (phase, repeat_index, total, round(float(remaining_s), 3))
                ),
            )

        assert wait_seen == [
                (ValidationPhase.WAIT_POSITION_SETTLE.value, 1, 1, 0.2),
                (ValidationPhase.WAIT_POSITION_SETTLE.value, 1, 1, 0.1),
                (ValidationPhase.WAIT_POSITION_SETTLE.value, 1, 1, 0.0),
                (ValidationPhase.WAIT_SAMPLE_DELAY.value, 1, 1, 0.1),
                (ValidationPhase.WAIT_SAMPLE_DELAY.value, 1, 1, 0.0),
            ]

    def test_fixed_section_repeatability_records_phase_sequence(self, request) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        tmp_root = repo_root / '.compile_check' / 'validation_workflow_phase'
        tmp_root.mkdir(parents=True, exist_ok=True)
        case_root = tmp_root / f'case_{int(time.time() * 1000)}'
        request.addfinalizer(lambda: shutil.rmtree(case_root, True))
        app_root = case_root / 'FRP_IPC'
        app_root.mkdir(parents=True, exist_ok=True)

        session = ValidationSession()
        workflow = ValidationWorkflow(
            recipe=Recipe(name='validation-phase'),
            calibration=CalibrationSnapshot(),
            runtime_state=RuntimeState.from_validation_session(session),
            gateway=StrictDeviceGateway(),
            run_repository=RunRepository(app_root_dir=app_root),
            validation_session=session,
        )
        request = FixedSectionRepeatabilityRequest(
            section_name='S1',
            metric_name='od_avg',
            repeat_count=2,
        )
        section_result = MeasureRow(
            idx=1,
            x_ui=0.0,
            x_abs=0.0,
            od_avg=100.0,
            od_dev=0.0,
            od_runout=0.0,
            od_round=0.0,
            id_avg=80.0,
            id_dev=0.0,
            id_runout=0.0,
            id_round=0.0,
            concentricity=0.0,
        )
        raw_points = [
            {
                'phase': 'SYNC',
                'ts': float(i),
                'theta_deg': float(i * 10),
                'bin': i,
                'od_mm': 100.0,
            }
            for i in range(6)
        ]
        windows = [
            {
                'window_index': 1,
                'window_role': 'SYNC',
                'point_count': len(raw_points),
                'theta_span_deg': 50.0,
            }
        ]

        progress_seen: list[tuple[int, int]] = []
        phase_seen: list[tuple[str, int, int]] = []
        with patch(
            'frp_workflow.validation_workflow.measure_current_position_section_capture',
            return_value=(section_result, raw_points, windows, {'cov': 1.0}),
        ) as capture_mock:
            rows, summary = workflow.run_fixed_section_repeatability(
                request,
                progress_callback=lambda index, total: progress_seen.append((index, total)),
                phase_callback=lambda event: phase_seen.append((event.phase, event.repeat_index, event.total)),
            )

        assert capture_mock.call_count == 2
        assert len(rows) == 2
        assert summary['count'] == 2
        assert progress_seen == [(1, 2), (2, 2)]
        assert workflow.current_phase == ValidationPhase.SAVE_RESULT

        phase_events = _phase_events(workflow)
        assert [event.phase for event in phase_events] == [
                ValidationPhase.PREPARE.value,
                ValidationPhase.BEFORE_CAPTURE.value,
                ValidationPhase.WAIT_POSITION_SETTLE.value,
                ValidationPhase.WAIT_SAMPLE_DELAY.value,
                ValidationPhase.CAPTURE.value,
                ValidationPhase.FIT_CALC.value,
                ValidationPhase.SAVE_RESULT.value,
                ValidationPhase.PREPARE.value,
                ValidationPhase.BEFORE_CAPTURE.value,
                ValidationPhase.WAIT_POSITION_SETTLE.value,
                ValidationPhase.WAIT_SAMPLE_DELAY.value,
                ValidationPhase.CAPTURE.value,
                ValidationPhase.FIT_CALC.value,
                ValidationPhase.SAVE_RESULT.value,
            ]
        assert [event.repeat_index for event in phase_events] == [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
        assert phase_seen == [(event.phase, event.repeat_index, event.total) for event in phase_events]
