from __future__ import annotations

import sys
import types
import unittest

_pymodbus = types.ModuleType("pymodbus")
_pymodbus_client = types.ModuleType("pymodbus.client")


class _FakeModbusTcpClient:
    pass


setattr(_pymodbus_client, "ModbusTcpClient", _FakeModbusTcpClient)
setattr(_pymodbus, "client", _pymodbus_client)
sys.modules.setdefault("pymodbus", _pymodbus)
sys.modules.setdefault("pymodbus.client", _pymodbus_client)

from core.models import Recipe
from domain.planning import RecipeSectionPlan, RecipeSectionPlanRow
from frp_workflow.autoflow_orchestrator import AutoFlowOrchestrator


class _FakeSectionLoopHost:
    _run_section_loop = AutoFlowOrchestrator._run_section_loop

    def __init__(self) -> None:
        self.recipe = Recipe(sample_delay_s=0.75)
        self.calls: list[tuple] = []

    def _raise_if_stop_requested(self) -> None:
        self.calls.append(("check_stop",))

    def _emit_progress(self, **kwargs) -> None:
        self.calls.append(("progress", int(kwargs["section_index"]), int(kwargs["section_total"])))

    def _emit_state(self, state: str, message: str) -> None:
        self.calls.append(("state", str(state), str(message)))

    def _move_linear_axes_to_targets(self, targets, *, context: str, strict: bool = True) -> None:
        self.calls.append(("move", dict(targets), str(context), bool(strict)))

    def _wait_before_section_capture(self, *, section_index: int, section_total: int, delay_s: float) -> None:
        self.calls.append(("wait", int(section_index), int(section_total), float(delay_s)))

    def _measure_section(self, **kwargs) -> None:
        self.calls.append(("measure", int(kwargs["section_index"]), float(kwargs["z_pos_mm"]), float(kwargs["x_abs"])))


class _FakeDelayHost:
    _wait_before_section_capture = AutoFlowOrchestrator._wait_before_section_capture

    def __init__(self) -> None:
        self.state_events: list[tuple[str, str]] = []
        self.stop_checks = 0

    def _emit_state(self, state: str, message: str) -> None:
        self.state_events.append((str(state), str(message)))

    def _raise_if_stop_requested(self) -> None:
        self.stop_checks += 1
        raise RuntimeError("stop requested")


class ProductionSampleDelayTest(unittest.TestCase):
    def test_section_loop_waits_before_measure(self) -> None:
        host = _FakeSectionLoopHost()
        section_plan = RecipeSectionPlan(
            positions_z=(12.5,),
            sections=(
                RecipeSectionPlanRow(
                    section_index=1,
                    z_od_disp=12.5,
                    z_id_disp=15.5,
                    ax0_abs=101.0,
                    ax1_abs=201.0,
                    ax4_abs=401.0,
                ),
            ),
        )

        host._run_section_loop(
            section_plan,
            centers_xyz=[],
            centers_xyz_id=[],
            concentricity_list=[],
        )

        sequence = [entry[0] for entry in host.calls if entry[0] in {"move", "wait", "measure"}]
        self.assertEqual(sequence, ["move", "wait", "measure"])
        self.assertIn(("wait", 1, 1, 0.75), host.calls)

    def test_wait_before_capture_can_be_stopped(self) -> None:
        host = _FakeDelayHost()

        with self.assertRaisesRegex(RuntimeError, "stop requested"):
            host._wait_before_section_capture(section_index=1, section_total=3, delay_s=1.0)

        self.assertGreaterEqual(host.stop_checks, 1)
        self.assertEqual(len(host.state_events), 1)
        self.assertEqual(host.state_events[0][0], "RUN")
        self.assertIn("wait sample delay", host.state_events[0][1])


if __name__ == "__main__":
    unittest.main()
