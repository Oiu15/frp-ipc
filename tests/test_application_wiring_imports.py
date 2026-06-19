from __future__ import annotations

"""Verify that the Phase 2 wiring modules are importable and expose
the expected public API.
"""

import inspect


class TestCompositionImports:
    def test_app_composition_is_dataclass(self) -> None:
        from application.composition import AppComposition

        assert hasattr(AppComposition, "__dataclass_fields__")
        assert "results_service" in AppComposition.__dataclass_fields__

    def test_build_app_composition_is_callable(self) -> None:
        from application.composition import build_app_composition

        assert callable(build_app_composition)


class TestEventWiringImports:
    def test_wire_ui_event_handlers_is_function(self) -> None:
        from application.event_wiring import wire_ui_event_handlers

        assert inspect.isfunction(wire_ui_event_handlers)


class TestControllerWiringImports:
    def test_wire_screen_controllers_is_function(self) -> None:
        from application.controller_wiring import wire_screen_controllers

        assert inspect.isfunction(wire_screen_controllers)


class TestAllModulesImportable:
    def test_all_wiring_modules_import_without_error(self) -> None:
        import application.composition  # noqa: F401
        import application.controller_wiring  # noqa: F401
        import application.event_wiring  # noqa: F401

        assert True
