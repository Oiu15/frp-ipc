from __future__ import annotations

"""批次6:主屏结果表 v2 列(构造冒烟 + 列一致性)。"""

import tkinter as tk
import types

import pytest

from ui.screens.main_screen import build_main_screen


class _FakePresenter:
    def __init__(self) -> None:
        self._widgets: dict[str, object] = {}
        self._vs: dict[str, object] = {}

    def __getattr__(self, name: str):
        # vend a StringVar for any presenter.*_var the screen references
        if name.endswith("_var"):
            v = tk.StringVar()
            object.__setattr__(self, name, v)
            return v
        raise AttributeError(name)

    def remember_widget(self, name, widget):
        self._widgets[name] = widget
        return widget

    def widget(self, name):
        return self._widgets.get(name)

    def remember_view_state(self, name, value):
        self._vs[name] = value
        return value

    def view_state(self, name, default=None):
        return self._vs.get(name, default)


def _fake_controller():
    names = [
        "handle_main_result_selection", "refresh_main_summary_panel",
        "start_measurement", "stop_measurement", "clear_measurement_results",
        "export_history_results", "open_serial_template_settings",
    ]
    return types.SimpleNamespace(**{n: (lambda *a, **k: None) for n in names})


def test_main_screen_builds_with_v2_columns():
    try:
        root = tk.Tk()
    except Exception:  # pragma: no cover - headless without Tk
        pytest.skip("no Tk display")
    root.withdraw()
    try:
        presenter = _FakePresenter()
        # build must not KeyError on headings/widths (covers every column)
        build_main_screen(ttk_parent(root), presenter=presenter, controller=_fake_controller(), ui=None)
        tree = presenter.widget("result_tree")
        assert tree is not None
        cols = tuple(str(c) for c in tree["columns"])
        # v2 columns present and table internally consistent (no overhang)
        for c in ("split_shift_deg", "coax_unreliable", "id_diam_v2", "id_round_v2", "concentricity_v2"):
            assert c in cols
        # v2 preset registered
        assert "id_diam_v2" in tuple(presenter.view_state("tree_displaycols_v2") or ())
        # legacy default view must NOT show v2 columns
        assert "id_diam_v2" not in tuple(presenter.view_state("tree_displaycols_sync") or ())
    finally:
        root.destroy()


def ttk_parent(root):
    from tkinter import ttk
    frame = ttk.Frame(root)
    frame.pack()
    return frame
