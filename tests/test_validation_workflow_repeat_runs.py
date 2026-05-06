import datetime as dt
import unittest
from pathlib import Path
from typing import NoReturn, Sequence

from application.state import CalibrationSnapshot, RunContext, RuntimeState, ValidationSession
from core.models import AxisComm, Recipe
from machine.device_gateway import ClChannel, ClReadResult, PollProfile, RegsRead
from frp_workflow.validation_workflow import ValidationWorkflow


class FakeGateway:
    def _unexpected_gateway_call(self, name: str) -> NoReturn:
        raise AssertionError(f'unexpected gateway call: {name}')

    def __getattr__(self, name: str):
        self._unexpected_gateway_call(name)

    def get_axis_copy(self, axis: int) -> AxisComm:
        self._unexpected_gateway_call("get_axis_copy")

    def movea_abs(self, axis: int, pos_abs: float, *, context: str = "MoveA") -> None:
        self._unexpected_gateway_call("movea_abs")

    def velmove(
        self,
        axis: int,
        velocity: float,
        *,
        acc: float = 80.0,
        dec: float = 80.0,
        jerk: float = 300.0,
    ) -> None:
        self._unexpected_gateway_call("velmove")

    def stop(self, axis: int) -> None:
        self._unexpected_gateway_call("stop")

    def halt(self, axis: int) -> None:
        self._unexpected_gateway_call("halt")

    def reset(self, axis: int) -> None:
        self._unexpected_gateway_call("reset")

    def enable(self, axis: int) -> None:
        self._unexpected_gateway_call("enable")

    def abort_motion(self, axes: Sequence[int] | None = None) -> None:
        self._unexpected_gateway_call("abort_motion")

    def apply_soft_limits_abs(
        self,
        axis: int,
        target_abs: float,
        *,
        strict: bool = False,
        context: str = "",
    ) -> float:
        self._unexpected_gateway_call("apply_soft_limits_abs")

    def read_regs_sync(self, d_addr: int, count: int, timeout_s: float = 0.35) -> RegsRead | None:
        self._unexpected_gateway_call("read_regs_sync")

    def read_axis_angle_deg_sync(self, axis: int = 3, timeout_s: float = 0.35) -> float | None:
        self._unexpected_gateway_call("read_axis_angle_deg_sync")

    def read_cl_sync(self, channel: ClChannel, *, timeout_s: float = 0.5) -> ClReadResult | None:
        self._unexpected_gateway_call("read_cl_sync")

    def set_plc_poll_profile(self, profile: PollProfile = "normal") -> None:
        self._unexpected_gateway_call("set_plc_poll_profile")

    def pulse_cmd_mask(self, axis: int, pulse_mask: int, pulse_ms: int = 120) -> None:
        self._unexpected_gateway_call("pulse_cmd_mask")

    def write_coil(self, coil_addr: int, value: int | bool) -> None:
        self._unexpected_gateway_call("write_coil")


class FakeRunRepository:
    def __init__(self) -> None:
        self._seq = 0
        self.prepared: list[tuple[str, str]] = []

    def prepare_run(self, recipe_name: str):
        from application.state import RunIdentity

        self._seq += 1
        serial = f'20260408-validation-{self._seq:03d}'
        run_id = f'run-{self._seq:03d}'
        started_at_ts = 1775606400.0 + float(self._seq)
        self.prepared.append((recipe_name, serial))
        return RunIdentity(serial=serial, run_id=run_id, started_at_ts=started_at_ts)

    def export_run(self, context: RunContext) -> str:
        raise AssertionError("unexpected repository call: export_run")

    def export_daily_summary(self, context: RunContext) -> None:
        raise AssertionError("unexpected repository call: export_daily_summary")


class FakeValidationRepository:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.exported_run_paths: list[Path] = []
        self.exported_summary_paths: list[Path] = []
        self.exported_statuses: list[str] = []

    def export_run(self, context):
        start_ts = float(context.started_at_ts if context.started_at_ts is not None else context.identity.started_at_ts)
        day_tag = dt.date.fromtimestamp(start_ts).strftime('%Y-%m-%d')
        run_dir = self.root / 'validation_exports' / day_tag / str(context.identity.serial)
        self.exported_run_paths.append(run_dir)
        self.exported_statuses.append(str(context.status))
        self.export_daily_summary(context)
        return str(run_dir)

    def export_daily_summary(self, context):
        start_ts = float(context.started_at_ts if context.started_at_ts is not None else context.identity.started_at_ts)
        day_tag = dt.date.fromtimestamp(start_ts).strftime('%Y-%m-%d')
        summary_path = self.root / 'validation_exports' / day_tag / 'summary.csv'
        self.exported_summary_paths.append(summary_path)


class ValidationWorkflowRepeatRunsTest(unittest.TestCase):
    def test_repeat_validation_runs_with_fake_gateway_and_repositories(self) -> None:
        run_repo = FakeRunRepository()
        export_repo = FakeValidationRepository(Path('/virtual/app_root'))
        exported_serials: list[str] = []

        for repeat_idx in range(1, 4):
            session = ValidationSession(
                standard_piece_id='STD-RING-001',
                validation_batch_id='VAL-BATCH-042',
                repeat_measurement_count=repeat_idx,
            )
            workflow = ValidationWorkflow(
                recipe=Recipe(name='validation-repeat'),
                calibration=CalibrationSnapshot(),
                runtime_state=RuntimeState.from_validation_session(session),
                gateway=FakeGateway(),
                run_repository=run_repo,
                validation_session=session,
            )

            identity = workflow.ensure_identity()
            workflow.record_state('PREP', f'prepare #{repeat_idx}')
            workflow.record_progress(step='baseline', index=repeat_idx, total=3, message='collecting')
            workflow.record_summary({'baseline_ok': True, 'repeat_idx': repeat_idx}, source='baseline')
            workflow.record_state('DONE', f'completed #{repeat_idx}')
            result = workflow.build_result(status='DONE', message=f'completed #{repeat_idx}', finished_at_ts=identity.started_at_ts + 5.0)
            export_ctx = workflow.build_export_context()
            run_dir = Path(export_repo.export_run(export_ctx))

            exported_serials.append(identity.serial)
            self.assertEqual(workflow.runtime_state.status, 'completed')
            self.assertEqual(result.status, 'DONE')
            self.assertEqual(result.identity, identity)
            self.assertEqual(result.standard_piece_id, 'STD-RING-001')
            self.assertEqual(result.validation_batch_id, 'VAL-BATCH-042')
            self.assertEqual(result.repeat_measurement_count, repeat_idx)
            self.assertEqual(session.summary_cache['repeat_idx'], repeat_idx)
            self.assertEqual(run_dir.parts[-3], 'validation_exports')
            self.assertEqual(run_dir.parts[-1], identity.serial)
            self.assertNotIn('exports', run_dir.parts[:-3])

        self.assertEqual(len(run_repo.prepared), 3)
        self.assertEqual(len(export_repo.exported_run_paths), 3)
        self.assertEqual(len(export_repo.exported_summary_paths), 3)
        self.assertEqual(export_repo.exported_statuses, ['DONE', 'DONE', 'DONE'])
        self.assertEqual(len(set(exported_serials)), 3)
        self.assertEqual(exported_serials, [
            '20260408-validation-001',
            '20260408-validation-002',
            '20260408-validation-003',
        ])
        for path in export_repo.exported_run_paths:
            self.assertIn('validation_exports', path.parts)
            self.assertNotIn('exports', path.parts[:-3])
        for path in export_repo.exported_summary_paths:
            self.assertEqual(path.name, 'summary.csv')
            self.assertIn('validation_exports', path.parts)


if __name__ == '__main__':
    unittest.main()
