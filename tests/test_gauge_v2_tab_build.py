from __future__ import annotations

"""批次5c-3:几何标定 V2 页构建冒烟 + controller 方法存在性守卫。

构建整页会触发每个 `presenter.get_var('tcal_*')`(缺变量即报错)与每个
`command=controller.<method>`(缺方法即报错),从而守住 UI 接线。
"""

import tkinter as tk
from tkinter import ttk
from typing import Any, cast

import pytest

from application.controllers.gauge_controller import GaugeController
from ui.presenters.gauge_presenter import GaugeScreenPresenter
from ui.screens.gauge_screen import _build_geometry_v2_tab


class _FakeView:
    def get_flag(self, name: str, default: bool = False) -> bool:
        return default


class _PermissiveController:
    """Any attribute resolves to a no-op callable (button command targets)."""

    def __getattr__(self, name: str) -> Any:
        return lambda *a, **k: None


_TCAL_METHODS = (
    "reload_tooling", "clear_tooling",
    "start_tcal_id_capture", "stop_tcal_id_capture", "add_tcal_dataset",
    "clear_tcal_datasets", "fit_tcal_id_pose", "apply_tcal_id_pose",
    "start_tcal_od_capture", "stop_tcal_od_capture",
    "compute_tcal_od_zero", "apply_tcal_od_zero", "capture_tcal_od_reference",
    "compute_tcal_od_psi", "apply_tcal_od_psi",
    "record_tcal_axis_low", "record_tcal_axis_high", "compute_tcal_axis", "apply_tcal_axis",
    "compute_tcal_chuck", "apply_tcal_chuck",
    "start_tcal_delta_capture", "stop_tcal_delta_capture",
    "compute_tcal_delta_reg", "apply_tcal_delta_reg",
    "run_tcal_selftest",
)


def test_gauge_controller_exposes_all_tcal_commands():
    for name in _TCAL_METHODS:
        assert callable(getattr(GaugeController, name, None)), f"GaugeController missing {name}"


def test_geometry_v2_tab_builds():
    prev_default_root = getattr(tk, "_default_root", None)
    try:
        root = tk.Tk()
    except Exception:  # pragma: no cover - headless without Tk
        pytest.skip("no Tk display")
    root.withdraw()
    try:
        presenter = GaugeScreenPresenter(cast(Any, _FakeView()), cast(Any, _PermissiveController()))
        presenter.ensure_vars(master=root)
        _build_geometry_v2_tab(ttk.Frame(root), presenter, _PermissiveController())
    finally:
        root.destroy()
        try:
            tk._default_root = prev_default_root  # type: ignore[attr-defined]
        except Exception:
            pass
