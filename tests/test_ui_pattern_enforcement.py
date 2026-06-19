from __future__ import annotations

import ast
from pathlib import Path
from typing import cast


ROOT = Path(__file__).resolve().parents[1]
DEVICE_GATEWAY = ROOT / "application" / "adapters" / "device_gateway.py"


def _source() -> str:
    return DEVICE_GATEWAY.read_text(encoding="utf-8-sig")


def _module() -> ast.Module:
    return ast.parse(_source())


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


def _class_method_names(class_name: str) -> list[str]:
    for node in _module().body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [
                item.name
                for item in node.body
                if isinstance(item, ast.FunctionDef)
            ]
    raise AssertionError(f"missing class {class_name}")


def test_screen_controller_fallback_allowlists_are_removed() -> None:
    source = _source()

    assert "_SCREEN_CONTROLLER_HOST_CALL_ALLOWLIST" not in source
    assert "_SCREEN_CONTROLLER_HOST_CALL_PREFIX_ALLOWLIST" not in source


def test_screen_presenter_fallback_allowlists_do_not_expand() -> None:
    assert _literal_strings("_SCREEN_PRESENTER_HOST_ATTR_ALLOWLIST") == ()
    assert _literal_strings("_SCREEN_PRESENTER_HOST_ATTR_PREFIX_ALLOWLIST") == ()
    assert _literal_strings("_SCREEN_PRESENTER_HOST_CALL_ALLOWLIST") == ()
    assert _literal_strings("_SCREEN_PRESENTER_HOST_CALL_PREFIX_ALLOWLIST") == ()


def test_screen_ui_context_fallback_allowlists_do_not_expand() -> None:
    assert _literal_strings("_SCREEN_UI_CONTEXT_ATTR_ALLOWLIST") == ()
    assert _literal_strings("_SCREEN_UI_CONTEXT_ATTR_PREFIX_ALLOWLIST") == ()


def test_generic_fallback_classes_do_not_gain_more_getattr_methods() -> None:
    assert _class_method_names("ScreenController").count("__getattr__") == 0
    assert _class_method_names("ScreenPresenter").count("__getattr__") == 1
    assert _class_method_names("ScreenUiContext").count("__getattr__") == 1
