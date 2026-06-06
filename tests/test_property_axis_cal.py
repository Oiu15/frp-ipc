"""Property tests for core/models.py — coordinate transform round-trip invariants.

AxisCal maps between servo feedback positions (abs) and displayed Z coordinates.
Every pair of inverse transforms must round-trip: applying one direction then the
other must return the original value.  Hypothesis generates valid machine-coordinate
values (±2 m envelope) across all axis/sign/offset combinations.

Subnormal floats are excluded because they lose precision when combined with
normal-magnitude offsets (e.g. 1e-216 + 1.0 → 1.0 in FP64).
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from core.models import AxisCal, UiCoord


# ---------------------------------------------------------------------------
# strategies — exclude subnormals to prevent precision-loss false positives
# ---------------------------------------------------------------------------


def _z_disp_values() -> st.SearchStrategy[float]:
    """Display-coordinate values within the machine's physical working range."""
    return st.floats(
        min_value=-2000.0, max_value=2000.0,
        allow_nan=False, allow_infinity=False, allow_subnormal=False,
    )


def _abs_positions() -> st.SearchStrategy[float]:
    """Absolute servo positions within the machine's physical working range."""
    return st.floats(
        min_value=-2000.0, max_value=2000.0,
        allow_nan=False, allow_infinity=False, allow_subnormal=False,
    )


def _offsets() -> st.SearchStrategy[float]:
    """Per-axis calibration offsets — small shifts around zero."""
    return st.floats(
        min_value=-500.0, max_value=500.0,
        allow_nan=False, allow_infinity=False, allow_subnormal=False,
    )


def _axes() -> st.SearchStrategy[int]:
    """Valid Z-mapping axes (AX3 is rotation-only, not part of Z mapping)."""
    return st.sampled_from([0, 1, 2, 4])


def _signs() -> st.SearchStrategy[int]:
    return st.sampled_from([-1, 1])


# ---------------------------------------------------------------------------
# AxisCal: z_disp ↔ abs round-trip
# ---------------------------------------------------------------------------


@given(
    axis=_axes(),
    sign=_signs(),
    off_ax=_offsets(),
    z_pos=_offsets(),
    z_disp=_z_disp_values(),
)
def test_axis_cal_disp_abs_roundtrip(
    axis: int, sign: int, off_ax: float, z_pos: float, z_disp: float
) -> None:
    """abs_to_z_disp(axis, z_disp_to_abs(axis, z_disp)) == z_disp"""
    cal = AxisCal(sign=sign, off_ax0=off_ax, off_ax1=off_ax, off_ax2=off_ax, off_ax4=off_ax, z_pos=z_pos)

    abs_pos = cal.z_disp_to_abs(axis, z_disp)
    result = cal.abs_to_z_disp(axis, abs_pos)

    # FP64 cannot round-trip values separated by >15 orders of magnitude
    # (e.g. 4e-216 + 1.0 → 1.0).  The mathematical identity holds; the
    # tolerance accounts for FP64 precision limits.
    assert result == pytest.approx(z_disp, abs=1e-12)


# ---------------------------------------------------------------------------
# AxisCal: z_raw ↔ abs round-trip
# ---------------------------------------------------------------------------


@given(
    axis=_axes(),
    sign=_signs(),
    off_ax=_offsets(),
    abs_pos=_abs_positions(),
)
def test_axis_cal_raw_abs_roundtrip(
    axis: int, sign: int, off_ax: float, abs_pos: float
) -> None:
    """z_raw_to_abs(axis, abs_to_z_raw(axis, abs)) == abs"""
    cal = AxisCal(sign=sign, off_ax0=off_ax, off_ax1=off_ax, off_ax2=off_ax, off_ax4=off_ax)

    z_raw = cal.abs_to_z_raw(axis, abs_pos)
    result = cal.z_raw_to_abs(axis, z_raw)

    assert result == pytest.approx(abs_pos, abs=1e-12)


# ---------------------------------------------------------------------------
# AxisCal: z_raw ↔ z_disp round-trip (IPC UI shift)
# ---------------------------------------------------------------------------


@given(
    z_raw=_z_disp_values(),
    z_pos=_offsets(),
)
def test_axis_cal_z_raw_disp_roundtrip(z_raw: float, z_pos: float) -> None:
    """z_raw_to_z_disp(z_disp_to_z_raw(x)) == x"""
    cal = AxisCal(z_pos=z_pos)

    disp = cal.z_raw_to_z_disp(z_raw)
    result = cal.z_disp_to_z_raw(disp)

    assert result == pytest.approx(z_raw, abs=1e-12)


# ---------------------------------------------------------------------------
# UiCoord: abs ↔ ui round-trip
# ---------------------------------------------------------------------------


@given(
    x_ui=st.floats(min_value=-5000.0, max_value=5000.0, allow_nan=False, allow_infinity=False, allow_subnormal=False),
    zero_abs=_abs_positions(),
    sign=_signs(),
)
def test_ui_coord_roundtrip(x_ui: float, zero_abs: float, sign: int) -> None:
    """abs_to_ui(ui_to_abs(x)) == x for any zero/sign combination."""
    coord = UiCoord(zero_abs=zero_abs, sign=sign)

    abs_val = coord.ui_to_abs(x_ui)
    result = coord.abs_to_ui(abs_val)

    assert result == pytest.approx(x_ui)
