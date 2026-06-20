from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from frp_workflow.steps.row_build_result import RowBuildResult


def test_row_build_result_keeps_row() -> None:
    row = {"section_idx": 1}

    result = RowBuildResult(row=row)

    assert result.row is row


def test_row_build_result_is_frozen() -> None:
    result = RowBuildResult(row={})

    with pytest.raises(FrozenInstanceError):
        setattr(result, "row", {"changed": True})
