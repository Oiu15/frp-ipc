"""Property tests for core/modbus_codec.py — FP64 and INT16 round-trip invariants.

Hypothesis generates edge-case inputs (NaN, Inf, -Inf, subnormals, extreme
integers) that hand-written tests rarely cover, catching precision bugs in
the Modbus register encode/decode chain.
"""

from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from core.modbus_codec import decode_fp64_le, decode_int16, encode_fp64_le, encode_int16


# ---------------------------------------------------------------------------
# FP64 round-trip
# ---------------------------------------------------------------------------


@given(x=st.floats())
def test_fp64_roundtrip(x: float) -> None:
    """decode(encode(x)) == x for every IEEE 754 float."""
    encoded = encode_fp64_le(x)
    assert len(encoded) == 4  # 4 Modbus registers
    assert all(0 <= r <= 0xFFFF for r in encoded)  # each register is 16-bit

    decoded = decode_fp64_le(encoded)

    if math.isnan(x):
        assert math.isnan(decoded)
    else:
        assert decoded == x


# ---------------------------------------------------------------------------
# INT16 round-trip
# ---------------------------------------------------------------------------


@given(v=st.integers(min_value=-32768, max_value=32767))
def test_int16_roundtrip(v: int) -> None:
    """decode_int16(encode_int16(v)) == v for all valid signed 16-bit integers."""
    encoded = encode_int16(v)
    assert 0 <= encoded <= 0xFFFF  # fits in one Modbus register

    decoded = decode_int16(encoded)
    assert decoded == v
