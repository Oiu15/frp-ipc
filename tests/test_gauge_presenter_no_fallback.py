from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from ui.presenters.gauge_presenter import GaugeScreenPresenter


ROOT = Path(__file__).resolve().parents[1]


class _FakeView:
    def __init__(self) -> None:
        self.variable = object()

    def get_var(self, name: str) -> Any:
        if name == "sample_var":
            return self.variable
        raise AttributeError(name)

    def get_flag(self, name: str, default: bool = False) -> bool:
        return True if name == "sample_enabled" else default

    def list_serial_ports(self) -> list[str]:
        return []


class _FakeController:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def list_validation_section_choices(self) -> list[str]:
        return ["S1", "S2"]

    def set_gauge_request_command(self, cmd: str) -> str:
        self.commands.append(cmd)
        return cmd


def test_gauge_presenter_source_has_no_dynamic_fallback() -> None:
    source = (ROOT / "ui" / "presenters" / "gauge_presenter.py").read_text(
        encoding="utf-8-sig"
    )

    for forbidden in (
        "def __getattr__",
        "getattr(self.controller",
        "getattr(controller",
        "getattr(presenter",
    ):
        assert forbidden not in source


def test_gauge_screen_uses_explicit_presenter_variable_access() -> None:
    source = (ROOT / "ui" / "screens" / "gauge_screen.py").read_text(
        encoding="utf-8-sig"
    )

    assert "presenter.get_var(" in source
    assert re.search(
        r"\bpresenter\.(?!get_var\b)[A-Za-z][A-Za-z0-9_]*_var\b",
        source,
    ) is None
    assert "getattr(presenter" not in source


def test_gauge_presenter_explicit_state_and_widget_access_preserves_identity() -> None:
    view = _FakeView()
    presenter = GaugeScreenPresenter(view, _FakeController())
    widget = object()

    assert presenter.get_var("sample_var") is view.variable
    assert presenter.get_flag("sample_enabled") is True
    assert presenter.get_flag("missing", False) is False
    assert presenter.remember_widget("sample_widget", widget) is widget
    assert presenter.widget("sample_widget") is widget


def test_gauge_presenter_uses_explicit_controller_methods() -> None:
    controller = _FakeController()
    presenter = GaugeScreenPresenter(_FakeView(), controller)

    assert presenter.validation_section_choices() == ["S1", "S2"]
    assert presenter.handle_request_command_changed("M0,1") == "M0,1"
    assert controller.commands == ["M0,1"]
