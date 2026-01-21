# ./core/modbus_codec.py
"""Small, pure helpers to encode/decode PLC Modbus register payloads.

Design goals:
- Pure functions (no IO, no app dependency)
- Explicit byte/word order

Conventions in this project:
- Modbus registers are 16-bit unsigned integers (0..65535).
- INT16 (Sign) is stored as a signed 16-bit integer in one register.
- FP64 (LREAL) is stored as 4 consecutive registers using **little-endian**
  ordering for the whole 8-byte payload.

  Implementation detail:
    bytes = pack('<4H', regs[0], regs[1], regs[2], regs[3])
    value = unpack('<d', bytes)

This matches the project's established "FP64 using le" convention.
"""

from __future__ import annotations

import struct
from typing import Iterable, List, Sequence


def decode_int16(reg: int) -> int:
    """Decode a signed INT16 from a Modbus register (0..65535)."""
    r = int(reg) & 0xFFFF
    return r - 0x10000 if r >= 0x8000 else r


def encode_int16(val: int) -> int:
    """Encode a signed INT16 into a Modbus register (0..65535)."""
    return int(val) & 0xFFFF


def decode_fp64_le(regs4: Sequence[int]) -> float:
    """Decode a float64 (FP64/LREAL) from 4 Modbus registers (little-endian).

    Args:
        regs4: length-4 sequence of 16-bit register values.

    Returns:
        Python float.
    """
    if len(regs4) != 4:
        raise ValueError(f"decode_fp64_le expects 4 regs, got {len(regs4)}")
    r0, r1, r2, r3 = (int(x) & 0xFFFF for x in regs4)
    b = struct.pack('<4H', r0, r1, r2, r3)
    return float(struct.unpack('<d', b)[0])


def encode_fp64_le(val: float) -> List[int]:
    """Encode a float64 (FP64/LREAL) into 4 Modbus registers (little-endian)."""
    b = struct.pack('<d', float(val))
    r0, r1, r2, r3 = struct.unpack('<4H', b)
    return [int(r0), int(r1), int(r2), int(r3)]
