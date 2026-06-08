from __future__ import annotations

from typing import Any

from tests.fakes import FakeVar

from application.host.main_view import HostMainViewMixin
from core.models import Recipe


class _FakeLabel:
    def __init__(self) -> None:
        self.text = ""

    def config(self, **kwargs: Any) -> None:
        self.text = str(kwargs.get("text", self.text))


class _FakeTree:
    def __init__(self, width: int) -> None:
        self.width = int(width)
        self.widths = {"idx": 90, "x_ui": 180, "od_fit_res": 120}
        self.configs: list[dict[str, Any]] = []

    def column(self, col: str, option: str | None = None, **kwargs: Any) -> Any:
        if kwargs:
            self.widths[col] = int(kwargs["width"])
            return None
        if option == "width":
            return self.widths[col]
        return None

    def winfo_width(self) -> int:
        return self.width

    def configure(self, **kwargs: Any) -> None:
        self.configs.append(dict(kwargs))


class _FakeMainHost(HostMainViewMixin):
    def __init__(self) -> None:
        self.recipe = Recipe(id_std_mm=50.0)
        self._id_samples = []
        self._run_serial = None
        self.pipe_sn_var = FakeVar()
        self.meas_seq_var = FakeVar()
        self.id_n_var = FakeVar()
        self.id_avg_var = FakeVar()
        self.id_dev_var = FakeVar()
        self.id_round_var = FakeVar()
        self.ui_meas_mode_var = FakeVar()
        self.max_id_dev_var = FakeVar()
        self.max_id_round_var = FakeVar()
        self.id_mean_var = FakeVar()
        self.id_dpp_var = FakeVar()
        self.id_range_var = FakeVar()
        self.id_slope_var = FakeVar()
        self.id_tilt_var = FakeVar()
        self.id_endoff_var = FakeVar()
        self.axis_dist_var = FakeVar()
        self.conc_max_var = FakeVar()
        self.axis_span_max_var = FakeVar()
        self.labels = {"lbl_od_std": _FakeLabel(), "lbl_id_std": _FakeLabel()}
        self.tree = _FakeTree(width=240)
        self.view_state = {
            "tree_displaycols_sync": ("idx", "x_ui", "od_fit_res"),
            "tree_displaycols_split": ("idx", "x_ui"),
            "tree_displaycols_od_only": ("idx", "od_fit_res"),
            "tree_column_widths": {"idx": 90, "x_ui": 180, "od_fit_res": 120},
            "tree_column_min_widths": {"idx": 50, "x_ui": 80, "od_fit_res": 80},
        }
        self.auto_clear_calls: list[bool] = []

    def _auto_clear_ui(self, preserve_run: bool = False) -> None:
        self.auto_clear_calls.append(bool(preserve_run))

    def _on_result_select(self, event: Any = None) -> str:
        return "selected"

    def _main_ui_widget(self, name: str) -> Any:
        if name == "result_tree":
            return self.tree
        return self.labels.get(name)

    def _main_view_state(self, name: str, default: Any = None) -> Any:
        return self.view_state.get(name, default)

    def after(self, ms: Any, func: Any | None = None, *args: Any) -> Any:
        if func is not None:
            return func(*args)
        return None


class TestHostMainViewMixin:
    def test_refresh_measurement_display_preserves_run_serial(self) -> None:
        host = _FakeMainHost()
        host._run_serial = "20260428-demo-007"

        host._refresh_measurement_display()

        assert host.auto_clear_calls == [True]
        assert host.pipe_sn_var.get() == "20260428-demo-007"
        assert host.meas_seq_var.get() == "007"

    def test_refresh_id_stats_computes_average_deviation_and_roundness(self) -> None:
        host = _FakeMainHost()
        host._id_samples = [49.0, 50.0, 52.0]

        host._refresh_id_stats()

        assert host.id_n_var.get() == "3"
        assert host.id_avg_var.get() == "50.333"
        assert host.id_dev_var.get() == "+0.333"
        assert host.id_round_var.get() == "3.000"

    def test_apply_main_ui_mode_sets_od_only_columns_and_placeholders(self) -> None:
        host = _FakeMainHost()
        host.recipe.disable_id_modbus = True

        host._apply_main_ui_mode()

        assert host._ui_get_meas_mode() == "OD_ONLY"
        assert host.tree.configs[-1]["displaycolumns"] == ("idx", "od_fit_res")
        assert "OD Only" in str(host.max_id_dev_var.get())
        assert "ID" in str(host.axis_dist_var.get())

    def test_resize_result_tree_columns_shrinks_to_available_width(self) -> None:
        host = _FakeMainHost()

        host._resize_result_tree_columns(host.tree, ("idx", "x_ui", "od_fit_res"))

        total = sum(host.tree.widths[col] for col in ("idx", "x_ui", "od_fit_res"))
        assert total <= 240
        assert host.tree.widths["x_ui"] >= 80
