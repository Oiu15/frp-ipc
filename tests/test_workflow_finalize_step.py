from __future__ import annotations

"""Unit test for FinalizeRunStep with a fake port."""

from frp_workflow.steps.finalize_run import FinalizeRunStep


class FakeRunSession:
    end_ts: float | None = None


class FakeMotion:
    def __init__(self) -> None:
        self.stops: list[int] = []
        self.aborts: int = 0

    def stop(self, axis: int) -> None:
        self.stops.append(axis)

    def abort_motion(self) -> None:
        self.aborts += 1


class FakeStopEvent:
    def __init__(self, is_set: bool = False) -> None:
        self._set = is_set

    def is_set(self) -> bool:
        return self._set


class FakeProductionWorkflow:
    def __init__(self) -> None:
        self.build_calls: list[dict] = []

    def build_run_result(self, **kwargs) -> dict:
        self.build_calls.append(dict(kwargs))
        return {"result": "ok", **kwargs}


class FakeFinalizePort:
    def __init__(self, *, stop_requested: bool = False, return_standby: bool = True) -> None:
        self.run_session = FakeRunSession()
        self.motion = FakeMotion()
        self._stop_event = FakeStopEvent(is_set=stop_requested)
        self._return_standby_after_stop = return_standby
        self.production_workflow = FakeProductionWorkflow()
        self.run_result = None
        self.internal_states: list[str] = []
        self.emitted_states: list[tuple[str, str]] = []
        self.standby_return_calls: int = 0

    def _set_internal_state(self, state: str) -> None:
        self.internal_states.append(state)

    def _emit_state(self, state: str, message: str) -> None:
        self.emitted_states.append((state, message))

    def _finalize_run_impl(self, status: str, message: str) -> None:
        self.run_session.end_ts = 12345.0
        self.motion.stop(3)
        if self._stop_event.is_set():
            self.motion.abort_motion()
            if self._return_standby_after_stop:
                self._return_to_standby_after_user_stop()
        if status == "DONE":
            self._set_internal_state("DONE")
        if self.production_workflow is not None:
            self.run_result = self.production_workflow.build_run_result(
                status=status, message=message, finished_at_ts=self.run_session.end_ts
            )
        self._emit_state(status, message)

    def _return_to_standby_after_user_stop(self) -> None:
        self.standby_return_calls += 1


class TestFinalizeRunStep:
    def test_step_name(self) -> None:
        step = FinalizeRunStep(FakeFinalizePort(), "DONE", "ok")
        assert step.name == "finalize_run"

    def test_done_status_sets_internal_state_and_emits(self) -> None:
        port = FakeFinalizePort()
        step = FinalizeRunStep(port, "DONE", "Measurement completed")

        step.execute()

        assert port.run_session.end_ts is not None
        assert port.motion.stops == [3]
        assert "DONE" in port.internal_states
        assert port.emitted_states[-1] == ("DONE", "Measurement completed")
        assert port.production_workflow.build_calls

    def test_stop_status_does_not_set_done(self) -> None:
        port = FakeFinalizePort(stop_requested=True)
        step = FinalizeRunStep(port, "STOP", "User stopped")

        step.execute()

        assert port.motion.aborts == 1
        assert port.standby_return_calls == 1
        assert "DONE" not in port.internal_states
        assert port.emitted_states[-1] == ("STOP", "User stopped")

    def test_error_status_sets_run_result_with_error(self) -> None:
        port = FakeFinalizePort()
        step = FinalizeRunStep(port, "ERR", "Something went wrong")

        step.execute()

        assert port.emitted_states[-1] == ("ERR", "Something went wrong")
        assert port.production_workflow.build_calls
        assert port.production_workflow.build_calls[-1]["status"] == "ERR"

    def test_delegates_to_impl_method(self) -> None:
        """Verify execute calls _finalize_run_impl on the port."""
        port = FakeFinalizePort()
        step = FinalizeRunStep(port, "DONE", "ok")

        step.execute()

        assert port.run_session.end_ts is not None
        assert len(port.motion.stops) == 1
