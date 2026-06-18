from __future__ import annotations

"""Contract test: verify that AppHost satisfies the application-layer port Protocols.

This test does NOT instantiate AppHost (which would require Tk).  Instead it
inspects the class's method and attribute surface to confirm that every port
method is present, which is what callers (mixins, services, tests) rely on.
"""

import inspect
from typing import get_type_hints

from application.app_host import AppHost
from application.host.ports import (
    EventPort,
    MotionPort,
    OperatorPort,
    RecipePort,
    RuntimeStatePort,
    TeachHost,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _public_names(protocol_type: type) -> set[str]:
    """Return the set of public attribute/method names defined by a Protocol."""
    return {
        name
        for name in dir(protocol_type)
        if not name.startswith("_")
        and callable(getattr(protocol_type, name, None))
    }


def _protocol_method_names(protocol_type: type) -> set[str]:
    """Return method names declared on a Protocol (excludes dunder, private)."""
    names: set[str] = set()
    for name, member in inspect.getmembers(protocol_type):
        if name.startswith("_"):
            continue
        if inspect.isfunction(member) or inspect.ismethod(member):
            names.add(name)
    # Also check annotations — some protocol methods may be listed only there
    try:
        hints = get_type_hints(protocol_type)
    except Exception:
        hints = {}
    for name in hints:
        if not name.startswith("_"):
            names.add(name)
    return names


def _host_has_attr(name: str) -> bool:
    """Check whether AppHost (or its parent classes) has a given attribute."""
    if hasattr(AppHost, name):
        return True
    # Some attributes (axis_cal, recipe) are set in __init__ and only appear
    # as class-level annotations — check those too.
    for klass in AppHost.__mro__:
        annotations = getattr(klass, "__annotations__", {})
        if name in annotations:
            return True
    return False


def _host_has_method(name: str) -> bool:
    """Stricter check: the name must be callable on the class."""
    return callable(getattr(AppHost, name, None))


# ---------------------------------------------------------------------------
# Port-specific expected members
# ---------------------------------------------------------------------------

_MOTION_METHODS = {
    "get_axis_copy",
    "movea_abs",
    "apply_soft_limits_abs",
    "_write_axis_params",
    "set_cmd_bits",
}

_MOTION_ATTRS = {"axis_cal"}

_OPERATOR_METHODS = {
    "show_error",
    "show_info",
    "show_warning",
    "ask_ok_cancel",
}

_RECIPE_METHODS = {
    "_recipe_ui_widget",
    "_recipe_apply_from_ui",
    "_get_selected_recipe_idx",
    "_ensure_recipe_section_plan",
    "_save_taught_section_to_recipe",
    "_refresh_recipe_table",
    "_refresh_length_info",
}

_RECIPE_ATTRS = {"recipe"}

_RUNTIME_STATE_METHODS = {"after"}

# EventPort is a lightweight placeholder — no required methods yet.
_EVENT_METHODS: set[str] = set()


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class TestMotionPortOnAppHost:
    def test_motion_methods_exist(self) -> None:
        missing = [m for m in _MOTION_METHODS if not _host_has_attr(m)]
        assert missing == [], f"AppHost missing MotionPort methods: {missing}"

    def test_motion_attrs_exist(self) -> None:
        missing = [a for a in _MOTION_ATTRS if not _host_has_attr(a)]
        assert missing == [], f"AppHost missing MotionPort attrs: {missing}"


class TestOperatorPortOnAppHost:
    def test_operator_methods_exist(self) -> None:
        missing = [m for m in _OPERATOR_METHODS if not _host_has_attr(m)]
        assert missing == [], f"AppHost missing OperatorPort methods: {missing}"


class TestRecipePortOnAppHost:
    def test_recipe_methods_exist(self) -> None:
        missing = [m for m in _RECIPE_METHODS if not _host_has_attr(m)]
        assert missing == [], f"AppHost missing RecipePort methods: {missing}"

    def test_recipe_attrs_exist(self) -> None:
        missing = [a for a in _RECIPE_ATTRS if not _host_has_attr(a)]
        assert missing == [], f"AppHost missing RecipePort attrs: {missing}"


class TestRuntimeStatePortOnAppHost:
    def test_runtime_state_methods_exist(self) -> None:
        missing = [m for m in _RUNTIME_STATE_METHODS if not _host_has_attr(m)]
        assert missing == [], f"AppHost missing RuntimeStatePort methods: {missing}"


class TestEventPortOnAppHost:
    def test_event_port_is_referenceable(self) -> None:
        """EventPort exists — it is intentionally a placeholder."""
        assert EventPort is not None


class TestTeachHostComposite:
    def test_apphost_satisfies_teach_host(self) -> None:
        """AppHost must satisfy all attributes/methods required by TeachHost.

        We don't use issubclass() because Protocol checks at runtime are
        unreliable.  Instead we walk every Protocol in TeachHost's MRO and
        verify that AppHost exposes each required name.
        """
        missing: list[str] = []

        for proto in inspect.getmro(TeachHost):
            if proto is TeachHost or proto is object:
                continue
            if not hasattr(proto, "_is_protocol") or not proto._is_protocol:
                continue
            for name in _protocol_method_names(proto):
                if not _host_has_attr(name):
                    missing.append(f"{proto.__name__}.{name}")

        assert missing == [], (
            f"AppHost missing TeachHost-required members: {missing}"
        )


# ---------------------------------------------------------------------------
# Ensure the port file is importable and all names are defined
# ---------------------------------------------------------------------------


class TestPortsModule:
    def test_all_exports_are_importable(self) -> None:
        from application.host.ports import (  # noqa: F811
            EventPort,
            MotionPort,
            OperatorPort,
            RecipePort,
            RuntimeStatePort,
            TeachHost,
        )

        assert MotionPort is not None
        assert OperatorPort is not None
        assert RecipePort is not None
        assert RuntimeStatePort is not None
        assert EventPort is not None
        assert TeachHost is not None
