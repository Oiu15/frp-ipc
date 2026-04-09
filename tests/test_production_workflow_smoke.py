import json
import shutil
import time
import unittest
from pathlib import Path

from application.state import CalibrationSnapshot, RunContext, RuntimeState
from core.models import MeasureRow, Recipe
from repositories.run_repository import RunRepository
from frp_workflow.production_workflow import ProductionWorkflow, ProductionWorkflowEventType


class FakeGateway:
    """Strict fake gateway for workflow boundary smoke tests."""

    def __getattr__(self, name: str):
        raise AssertionError(f"unexpected gateway call: {name}")


class ProductionWorkflowSmokeTest(unittest.TestCase):
    def test_smoke_done_flow_and_export(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        tmp_root = repo_root / ".compile_check" / "workflow_smoke"
        tmp_root.mkdir(parents=True, exist_ok=True)
        case_root = tmp_root / f"case_{int(time.time() * 1000)}"
        self.addCleanup(shutil.rmtree, case_root, True)
        app_root = case_root / "FRP_IPC"
        app_root.mkdir(parents=True, exist_ok=True)

        repo = RunRepository(app_root_dir=app_root)
        recipe = Recipe(
                name="smoke",
                section_count=1,
                section_pos_z=[100.0],
                len_enable=False,
            )
        calibration = CalibrationSnapshot()
        runtime = RuntimeState()
        workflow = ProductionWorkflow(
                recipe=recipe,
                calibration=calibration,
                runtime_state=runtime,
                gateway=FakeGateway(),
                run_repository=repo,
            )

        identity = workflow.ensure_identity()
        workflow.record_state("PREP", "prepare")
        workflow.record_state("RUN", "running")
        workflow.record_progress(
                section_index=1,
                section_total=1,
                z_pos_mm=100.0,
                ax0_abs=200.0,
            )
        workflow.record_length({"len_mm": 1680.0, "ok": True})
        cov_payload = {
                "section_idx": 1,
                "cov": 0.98,
                "miss": 2,
                "max_gap_deg": 5.0,
                "revs": 1.1,
                "elapsed": 1.25,
                "reason": "COV",
            }
        workflow.record_coverage(cov_payload)
        row = MeasureRow(
                idx=1,
                x_ui=100.0,
                x_abs=200.0,
                od_avg=187.31,
                od_dev=0.01,
                od_runout=0.02,
                od_round=0.03,
                id_avg=152.70,
                id_dev=0.01,
                id_runout=0.02,
                id_round=0.03,
                concentricity=0.04,
            )
        workflow.record_row(row)
        workflow.record_raw_points([
                {
                    "section_idx": 1,
                    "z_pos_mm": 100.0,
                    "sample_idx": 0,
                    "ts": 123.456,
                    "theta_deg": 0.0,
                    "bin": 0,
                    "phase": "od",
                    "od_mm": 187.3,
                    "id_mm": 152.7,
                    "cl_cnt": 1,
                    "raw_od": 187.3,
                    "raw_id": 152.7,
                }
            ])
        workflow.record_summary(
                {
                    "straight_od_mm": 0.12,
                    "straight_id_mm": 0.08,
                    "summary_ok": True,
                },
                source="straightness",
            )
        workflow.record_state("DONE", "completed")
        result = workflow.build_run_result(
                status="DONE",
                message="completed",
                finished_at_ts=time.time(),
            )

        ctx = RunContext(
                identity=identity,
                recipe=recipe,
                calibration=calibration,
                rows=list(result.rows),
                raw_points=list(workflow.raw_points.points),
                section_coverage={1: cov_payload},
                length_result=result.length_result,
                summary=dict(result.summary),
                finished_at_ts=result.finished_at_ts,
                status=result.status,
            )
        run_dir = Path(repo.export_run(ctx))

        self.assertEqual(runtime.status, "completed")
        self.assertEqual(result.status, "DONE")
        self.assertEqual(result.identity, identity)
        self.assertEqual(
                [event.type for event in workflow.events],
                [
                    ProductionWorkflowEventType.STATE,
                    ProductionWorkflowEventType.STATE,
                    ProductionWorkflowEventType.PROGRESS,
                    ProductionWorkflowEventType.LENGTH,
                    ProductionWorkflowEventType.COVERAGE,
                    ProductionWorkflowEventType.ROW,
                    ProductionWorkflowEventType.RAW_POINTS,
                    ProductionWorkflowEventType.SUMMARY,
                    ProductionWorkflowEventType.STATE,
                ],
            )
        self.assertEqual(run_dir.name, identity.serial)
        self.assertTrue((run_dir / "section_results.csv").exists())
        self.assertTrue((run_dir / "raw_points.csv").exists())
        self.assertTrue((run_dir / "meta.json").exists())
        self.assertTrue((run_dir.parent / "summary.csv").exists())

        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["serial"], identity.serial)
        self.assertEqual(meta["run_id"], identity.run_id)
        self.assertEqual(Path(meta["exports"]["meta_json"]).name, "meta.json")


if __name__ == "__main__":
    unittest.main()
