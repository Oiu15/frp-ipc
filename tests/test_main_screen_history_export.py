import inspect

from ui.screens.main_screen import build_main_screen


def test_main_screen_exposes_manual_history_export_button() -> None:
    source = inspect.getsource(build_main_screen)

    assert "导出结果" in source
    assert "export_history_results" in source
