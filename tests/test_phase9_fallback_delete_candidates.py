from __future__ import annotations

import ast
from pathlib import Path
from typing import cast


ROOT = Path(__file__).resolve().parents[1]
DEVICE_GATEWAY = ROOT / "application" / "adapters" / "device_gateway.py"


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8-sig")


def _module() -> ast.Module:
    return ast.parse(DEVICE_GATEWAY.read_text(encoding="utf-8-sig"))


def _literal_strings(name: str) -> tuple[str, ...]:
    for node in _module().body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return _literal_string_value(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return _literal_string_value(node.value)
    raise AssertionError(f"missing constant {name}")


def _literal_string_value(node: ast.expr | None) -> tuple[str, ...]:
    if node is None:
        raise AssertionError("missing assigned value")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "set":
        return ()
    value = ast.literal_eval(node)
    if isinstance(value, set):
        return tuple(sorted(cast(set[str], value)))
    if isinstance(value, tuple):
        return cast(tuple[str, ...], value)
    raise AssertionError(f"unsupported literal {ast.dump(node)}")


def test_generic_fallback_allowlists_are_empty_before_delete_audit() -> None:
    for name in (
        "_SCREEN_PRESENTER_HOST_ATTR_ALLOWLIST",
        "_SCREEN_PRESENTER_HOST_ATTR_PREFIX_ALLOWLIST",
        "_SCREEN_PRESENTER_HOST_CALL_ALLOWLIST",
        "_SCREEN_PRESENTER_HOST_CALL_PREFIX_ALLOWLIST",
        "_SCREEN_UI_CONTEXT_ATTR_ALLOWLIST",
        "_SCREEN_UI_CONTEXT_ATTR_PREFIX_ALLOWLIST",
    ):
        assert _literal_strings(name) == ()

    source = DEVICE_GATEWAY.read_text(encoding="utf-8-sig")
    assert "_SCREEN_CONTROLLER_HOST_CALL_ALLOWLIST" not in source
    assert "_SCREEN_CONTROLLER_HOST_CALL_PREFIX_ALLOWLIST" not in source


def test_remaining_getattr_methods_still_exist_after_controller_fallback_deletion() -> None:
    source = DEVICE_GATEWAY.read_text(encoding="utf-8-sig")

    assert "class ScreenController" in source
    assert "class ScreenPresenter" in source
    assert "class ScreenUiContext" in source
    assert source.count("def __getattr__(self, name: str) -> Any:") >= 2


def test_migrated_screens_do_not_depend_on_generic_fallback_tokens() -> None:
    for path in (
        "ui/screens/recipe_screen.py",
        "ui/screens/key_test_screen.py",
        "ui/screens/axis_cal_screen.py",
        "ui/screens/validation_screen.py",
        "ui/screens/gauge_screen.py",
        "ui/screens/main_screen.py",
        "ui/screens/axis_screen.py",
    ):
        source = _read(path)
        for forbidden in (
            "getattr(controller",
            "getattr(presenter",
            "getattr(ui",
            "controller._host",
            "presenter._host",
            "ui._host",
        ):
            assert forbidden not in source, f"{forbidden} found in {path}"


def test_screen_ui_context_remaining_wiring_is_inventory_only() -> None:
    source = _read("application/host/ui.py")

    assert "build_axis_screen(tab_axis, presenter=self._axis_screen_presenter, controller=self.axis_controller, ui=self._screen_ui_context)" in source
    assert "build_recipe_screen(tab_recipe, presenter=self._recipe_screen_presenter, controller=self.recipe_controller, ui=self._screen_ui_context)" in source
    assert "build_gauge_screen(tab_gauge, presenter=self._gauge_screen_presenter, controller=self.gauge_controller, ui=self._screen_ui_context" not in source
    assert "build_main_screen(tab_main, presenter=self.main_ui, controller=self.main_controller, ui=self._screen_ui_context" not in source
