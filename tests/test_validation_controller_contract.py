from __future__ import annotations

from typing import Any

from application.controllers.validation_controller import ValidationController


class _FakeValidationHost:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.stop_calls = 0
        self.feedback: list[dict[str, Any]] = []
        self.section_choices = ["1: 100.000", "2: 200.000"]

    def list_validation_section_choices(self) -> list[str]:
        return list(self.section_choices)

    def start_validation_run(self, **kwargs: Any) -> str:
        self.calls.append(dict(kwargs))
        return "started"

    def stop_validation_run(self) -> str:
        self.stop_calls += 1
        return "stopped"

    def _set_validation_feedback(self, **kwargs: Any) -> None:
        self.feedback.append(dict(kwargs))


def test_validation_controller_lists_section_choices() -> None:
    host = _FakeValidationHost()
    controller = ValidationController(host)

    assert controller.list_validation_section_choices() == ["1: 100.000", "2: 200.000"]


def test_validation_controller_forwards_normalized_run_options() -> None:
    host = _FakeValidationHost()
    controller = ValidationController(host)

    result = controller.start_validation_run(
        section_name=" S1 ",
        metric_name="od_avg",
        repeat_count="2",
        reclamp_enabled="true",
        rotation_stop_before_measure=True,
        release_settle_s="0.25",
        clamp_settle_s="0.5",
        position_settle_s="0.75",
        sample_delay_s="0.125",
        validation_ax3_speed_dps="45",
        move_enabled="true",
        move_channel="id_channel",
        move_away_delta_mm="12.5",
        move_scenario="switch_and_return",
        move_from_section_index="1: 100.000",
        move_target_section_index="2: 200.000",
        move_return_section_index="1: 100.000",
    )

    assert result == "started"
    assert host.calls == [
        {
            "section_name": "S1",
            "metric_name": "od_avg",
            "repeat_count": 2,
            "reclamp_between_repeats": False,
            "reclamp_enabled": True,
            "rotation_stop_before_measure": True,
            "release_settle_s": 0.25,
            "clamp_settle_s": 0.5,
            "position_settle_s": 0.75,
            "sample_delay_s": 0.125,
            "validation_ax3_speed_dps": 45.0,
            "move_enabled": True,
            "move_channel": "id_channel",
            "move_away_delta_mm": 12.5,
            "move_scenario": "switch_and_return",
            "move_from_section_index": 1,
            "move_target_section_index": 2,
            "move_return_section_index": 1,
        }
    ]


def test_validation_controller_reports_invalid_input_to_host_feedback() -> None:
    host = _FakeValidationHost()
    controller = ValidationController(host)

    result = controller.start_validation_run(
        section_name="S1",
        metric_name="od_avg",
        repeat_count="",
    )

    assert result is None
    assert host.calls == []
    assert host.feedback == [
        {
            "status": "ERR",
            "result": "",
            "error": "repeat_count cannot be empty",
            "export_path": "",
        }
    ]


def test_validation_controller_forwards_stop() -> None:
    host = _FakeValidationHost()
    controller = ValidationController(host)

    assert controller.stop_validation_run() == "stopped"
    assert host.stop_calls == 1
