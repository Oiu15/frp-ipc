from __future__ import annotations

import ast
from pathlib import Path
from typing import cast


ROOT = Path(__file__).resolve().parents[1]
DEVICE_GATEWAY = ROOT / "application" / "adapters" / "device_gateway.py"


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


def test_migrated_screen_command_prefixes_are_removed_from_controller_allowlist() -> None:
    source = DEVICE_GATEWAY.read_text(encoding="utf-8-sig")

    assert "_SCREEN_CONTROLLER_HOST_CALL_ALLOWLIST" not in source
    assert "_SCREEN_CONTROLLER_HOST_CALL_PREFIX_ALLOWLIST" not in source


def test_migrated_screen_state_prefixes_are_removed_from_presenter_and_ui_allowlists() -> None:
    source = DEVICE_GATEWAY.read_text(encoding="utf-8-sig")
    assert "_SCREEN_PRESENTER_HOST_ATTR_ALLOWLIST" not in source
    assert "_SCREEN_PRESENTER_HOST_ATTR_PREFIX_ALLOWLIST" not in source
    assert "_SCREEN_PRESENTER_HOST_CALL_ALLOWLIST" not in source
    assert "_SCREEN_PRESENTER_HOST_CALL_PREFIX_ALLOWLIST" not in source
    assert "_SCREEN_UI_CONTEXT_ATTR_ALLOWLIST" not in source
    assert "_SCREEN_UI_CONTEXT_ATTR_PREFIX_ALLOWLIST" not in source


def test_generic_getattr_methods_are_removed() -> None:
    source = DEVICE_GATEWAY.read_text(encoding="utf-8-sig")

    assert "class ScreenController" in source
    assert "class ScreenPresenter" in source
    assert "class ScreenUiContext" in source
    assert "def __getattr__(self, name: str) -> Any:" not in source
