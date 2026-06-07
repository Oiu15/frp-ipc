"""Tests for drivers/gauge_driver.py pure protocol-parsing functions.

These functions are stateless and can be tested without serial hardware.
"""

from __future__ import annotations

import pytest

from drivers.gauge_driver import _is_judge_token, parse_gauge_line


class TestIsJudgeToken:
    """_is_judge_token — discriminate judge/enum tokens from numeric values."""

    @pytest.mark.parametrize(
        ("s", "expected"),
        [
            ("GO", True),
            ("HI", True),
            ("LO", True),
            ("HH", True),
            ("LL", True),
            ("NG", True),
            ("go", True),
            (" go ", True),
            ("12.345", False),
            ("+187.3", False),
            ("-0.01", False),
            ("", False),
            ("  ", False),
        ],
    )
    def test_is_judge_token(self, s: str, expected: bool) -> None:
        assert _is_judge_token(s) is expected


class TestParseGaugeLine:
    """parse_gauge_line — M0/M1/M2 frame parsing and value extraction."""

    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            # M1: OUT1 only
            ("M1,+187.3,GO", (187.3, "GO", None, "UNK")),
            ("M1,+152.7", (152.7, "UNK", None, "UNK")),
            ("M1,-0.05,LL", (-0.05, "LL", None, "UNK")),
            # M0: OUT1 + OUT2
            ("M0,+187.3,GO,+187.1,GO", (187.3, "GO", 187.1, "GO")),
            ("M0,100.0,HI,80.0,LO", (100.0, "HI", 80.0, "LO")),
            ("M0,100.0,80.0", (100.0, "UNK", 80.0, "UNK")),
            # M2: same as M1 (no OUT2)
            ("M2,+50.0,GO", (50.0, "GO", None, "UNK")),
            # invalid / edge
            ("garbage", None),
            ("", None),
            ("  ", None),
            ("M1", None),
            ("M1,not_a_number", None),
            ("M0,1.0,GO", None),  # M0 requires OUT2
        ],
    )
    def test_parse_gauge_line(self, line: str, expected: tuple | None) -> None:
        result = parse_gauge_line(line)
        if expected is None:
            assert result is None
        else:
            assert result is not None
            assert result == pytest.approx(expected, nan_ok=True)
