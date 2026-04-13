from __future__ import annotations

import unittest
from unittest.mock import patch

from application.app_host import AppHost


class _FakeModeMachine:
    def __init__(self, events: list[tuple]) -> None:
        self._events = events
        self.current_mode_kind = "production"

    def enter_validation(self) -> None:
        self._events.append(("enter_validation",))
        self.current_mode_kind = "validation"

    def sync_current_mode_state(self) -> None:
        self._events.append(("sync_current_mode_state",))


class _FakeValidationRepository:
    def export_fixed_section_repeatability(self, **kwargs) -> str:
        return "validation-export-dir"


class _WorkflowSuccess:
    def __init__(self, *args, **kwargs) -> None:
        self.validation_session = kwargs.get("validation_session")
        self.fixed_section_repeat_captures = []

    def run_fixed_section_repeatability(self, request, **kwargs):
        return [], {"count": 1, "mean": 1.23, "std": 0.0}

    def build_export_context(self):
        return {}


class _WorkflowTimeout(_WorkflowSuccess):
    def run_fixed_section_repeatability(self, request, **kwargs):
        raise TimeoutError("AX0 in-position timeout")


class _FakeAutoThread:
    def __init__(self, *, alive: bool) -> None:
        self._alive = bool(alive)
        self.stop_calls = 0

    def is_alive(self) -> bool:
        return self._alive

    def stop(self) -> None:
        self.stop_calls += 1


class _FakeValidationHost:
    _current_mode_kind_name = AppHost._current_mode_kind_name
    _is_auto_thread_alive = AppHost._is_auto_thread_alive
    _prepare_validation_debug_run = AppHost._prepare_validation_debug_run
    _cleanup_validation_debug_run = AppHost._cleanup_validation_debug_run
    _finish_validation_debug_run_ui = AppHost._finish_validation_debug_run_ui
    start_fixed_section_repeatability_debug = AppHost.start_fixed_section_repeatability_debug
    _stop_measurement_impl = AppHost._stop_measurement_impl

    def __init__(
        self,
        *,
        recipe_exception: Exception | None = None,
        after_exception: Exception | None = None,
        auto_thread: _FakeAutoThread | None = None,
    ) -> None:
        self.events: list[tuple] = []
        self.feedback: list[dict] = []
        self.start_button_states: list[bool] = []
        self.move_positions: list[dict] = []
        self._plc_poll_profile_req = "sampling"
        self._validation_debug_running = False
        self._validation_thread = None
        self.validation_session = None
        self._auto_thread = auto_thread
        self._recipe_exception = recipe_exception
        self._after_exception = after_exception
        self.mode_machine = _FakeModeMachine(self.events)

    def set_plc_poll_profile(self, profile: str = "normal", *, caller: str | None = None) -> None:
        self.events.append(("set_plc_poll_profile", str(profile), str(caller or "")))
        self._plc_poll_profile_req = str(profile)

    def _validation_recipe_snapshot_from_ui(self):
        if self._recipe_exception is not None:
            raise self._recipe_exception
        return object()

    def get_calibration_snapshot(self):
        return object()

    def _make_run_repository(self):
        return object()

    def _make_validation_repository(self):
        return _FakeValidationRepository()

    def _set_validation_debug_feedback(self, **kwargs) -> None:
        self.feedback.append(dict(kwargs))

    def _set_validation_debug_start_button_state(self, enabled: bool) -> None:
        self.start_button_states.append(bool(enabled))

    def _set_validation_debug_move_position(self, **kwargs) -> None:
        self.move_positions.append(dict(kwargs))

    def update_idletasks(self) -> None:
        self.events.append(("update_idletasks",))

    def after(self, delay_ms: int, callback) -> None:
        self.events.append(("after", int(delay_ms)))
        if self._after_exception is not None:
            raise self._after_exception
        callback()

    def abort_motion(self) -> None:
        self.events.append(("abort_motion",))


def _thread_factory(events: list[tuple], *, run_target: bool):
    def _build_thread(*args, **kwargs):
        target = kwargs.get("target")
        name = kwargs.get("name")
        events.append(("thread_init", str(name or "")))

        class _Thread:
            def start(self_nonlocal) -> None:
                events.append(("thread_start", str(name or "")))
                if run_target and callable(target):
                    target()

        return _Thread()

    return _build_thread


class AppHostValidationPollingTest(unittest.TestCase):
    def _start_validation(
        self,
        host: _FakeValidationHost,
        *,
        workflow_cls,
        run_thread_target: bool,
    ) -> None:
        with patch("application.app_host.ValidationWorkflow", new=workflow_cls):
            with patch("application.app_host.threading.Thread", new=_thread_factory(host.events, run_target=run_thread_target)):
                host.start_fixed_section_repeatability_debug(
                    section_name="S1",
                    metric_name="od_avg",
                    repeat_count=1,
                    move_enabled=True,
                    move_channel="od_channel",
                    move_away_delta_mm=12.5,
                    move_scenario="distance_round_trip",
                    move_from_section_index=1,
                    move_target_section_index=1,
                    move_return_section_index=1,
                )

    def test_validation_start_forces_normal_before_thread_starts(self) -> None:
        host = _FakeValidationHost()

        self._start_validation(host, workflow_cls=_WorkflowSuccess, run_thread_target=False)

        set_idx = host.events.index(("set_plc_poll_profile", "normal", "validation_debug_enter"))
        thread_idx = host.events.index(("thread_start", "validation-fixed-section-repeatability"))
        self.assertLess(set_idx, thread_idx)
        self.assertEqual(host._plc_poll_profile_req, "normal")
        self.assertIsNotNone(host._validation_thread)

    def test_validation_success_cleanup_restores_normal_polling(self) -> None:
        host = _FakeValidationHost()

        self._start_validation(host, workflow_cls=_WorkflowSuccess, run_thread_target=True)

        self.assertEqual(
            [event for event in host.events if event[0] == "set_plc_poll_profile"],
            [
                ("set_plc_poll_profile", "normal", "validation_debug_enter"),
                ("set_plc_poll_profile", "normal", "validation_debug_exit"),
            ],
        )
        self.assertFalse(host._validation_debug_running)
        self.assertIsNone(host._validation_thread)
        self.assertEqual(host.start_button_states, [False, True])
        self.assertEqual(host.feedback[-1]["status"], "DONE")

    def test_validation_timeout_cleanup_restores_normal_polling(self) -> None:
        host = _FakeValidationHost()

        self._start_validation(host, workflow_cls=_WorkflowTimeout, run_thread_target=True)

        self.assertEqual(
            [event for event in host.events if event[0] == "set_plc_poll_profile"],
            [
                ("set_plc_poll_profile", "normal", "validation_debug_enter"),
                ("set_plc_poll_profile", "normal", "validation_debug_exit"),
            ],
        )
        self.assertFalse(host._validation_debug_running)
        self.assertIsNone(host._validation_thread)
        self.assertEqual(host.feedback[-1]["status"], "ERR")
        self.assertIn("timeout", host.feedback[-1]["error"].lower())

    def test_validation_startup_exception_still_restores_normal_polling(self) -> None:
        host = _FakeValidationHost(recipe_exception=RuntimeError("recipe snapshot failed"))

        self._start_validation(host, workflow_cls=_WorkflowSuccess, run_thread_target=False)

        self.assertEqual(
            [event for event in host.events if event[0] == "set_plc_poll_profile"],
            [
                ("set_plc_poll_profile", "normal", "validation_debug_enter"),
                ("set_plc_poll_profile", "normal", "validation_debug_exit"),
            ],
        )
        self.assertFalse(host._validation_debug_running)
        self.assertIsNone(host._validation_thread)
        self.assertEqual(host.feedback[-1]["status"], "ERR")
        self.assertIn("recipe snapshot failed", host.feedback[-1]["error"])

    def test_validation_worker_exception_cleanup_runs_even_when_after_fails(self) -> None:
        host = _FakeValidationHost(after_exception=RuntimeError("ui loop unavailable"))

        self._start_validation(host, workflow_cls=_WorkflowTimeout, run_thread_target=True)

        self.assertEqual(
            [event for event in host.events if event[0] == "set_plc_poll_profile"],
            [
                ("set_plc_poll_profile", "normal", "validation_debug_enter"),
                ("set_plc_poll_profile", "normal", "validation_debug_exit"),
            ],
        )
        self.assertFalse(host._validation_debug_running)
        self.assertIsNone(host._validation_thread)

    def test_stop_measurement_requests_normal_polling_before_abort(self) -> None:
        auto_thread = _FakeAutoThread(alive=True)
        host = _FakeValidationHost(auto_thread=auto_thread)

        host._stop_measurement_impl()

        self.assertEqual(auto_thread.stop_calls, 1)
        self.assertEqual(
            host.events,
            [
                ("set_plc_poll_profile", "normal", "stop_measurement"),
                ("abort_motion",),
            ],
        )
        self.assertEqual(host._plc_poll_profile_req, "normal")


if __name__ == "__main__":
    unittest.main()
