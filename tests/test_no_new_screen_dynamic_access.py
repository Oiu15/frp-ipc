from __future__ import annotations

"""Guard: recipe_screen.py must not use dynamic proxy patterns on controller,
presenter, or ui after Phase 3 migration.

This test reads the source text and checks for forbidden patterns.
"""

from pathlib import Path


_RECIPE_SCREEN = Path(__file__).resolve().parents[1] / "ui" / "screens" / "recipe_screen.py"


def _source() -> str:
    return _RECIPE_SCREEN.read_text(encoding="utf-8-sig")


class TestNoDynamicAccessOnRecipeScreen:
    def test_no_getattr_on_controller(self) -> None:
        source = _source()
        offenders: list[str] = []
        for line in source.splitlines():
            stripped = line.strip()
            if "getattr(controller" in stripped or "getattr( controller" in stripped:
                offenders.append(stripped[:120])
        assert offenders == [], f"getattr(controller, ...) found: {offenders}"

    def test_no_controller_host_access(self) -> None:
        source = _source()
        # controller._host or controller.host_app should not appear
        for forbidden in ("controller._host", "controller.host_app"):
            assert forbidden not in source, f"{forbidden} found in recipe_screen.py"

    def test_no_getattr_on_presenter_for_methods(self) -> None:
        source = _source()
        offenders: list[str] = []
        for line in source.splitlines():
            stripped = line.strip()
            # Allow getattr(presenter, varname, default) for Tk var lookups
            # but flag getattr(presenter, method_name) usage
            if "getattr(presenter" in stripped:
                # These are the allowed patterns — presenter attr fallbacks
                if any(
                    allowed in stripped
                    for allowed in (
                        "_recipe_sash_inited",
                        "len_enable_var",
                        "len_z_low_approach_var",
                        "len_low_search_dist_var",
                        "len_high_search_dist_var",
                        "len_search_vel_var",
                        "len_search_timeout_var",
                        "len_tol_var",
                        "len_high_margin_var",
                        "pipe_len_var",
                        "_refresh_length_info",
                    )
                ):
                    continue
                offenders.append(stripped[:120])
        assert offenders == [], f"unexpected getattr(presenter, ...) found: {offenders}"

    def test_no_getattr_on_ui_for_host_access(self) -> None:
        source = _source()
        for forbidden in ("getattr(ui", "getattr( ui"):
            assert forbidden not in source, f"{forbidden} found in recipe_screen.py"
