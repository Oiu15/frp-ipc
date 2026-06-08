from __future__ import annotations

import importlib.util
import typing
from collections.abc import Sequence

from domain import protocols, validation_models
from domain.state import CalibrationSnapshot, RunContext, RunIdentity, ValidationExportContext
from domain.validation_models import (
    FixedSectionRepeatCapture,
    FixedSectionRepeatabilityRequest,
    FixedSectionRepeatRow,
)


class TestDomainProtocols:
    def test_domain_boundary_modules_have_effective_docstrings(self) -> None:
        assert protocols.__doc__
        assert validation_models.__doc__

    def test_protocol_annotations_resolve(self) -> None:
        for protocol in (
            protocols.RunRepositoryProtocol,
            protocols.ValidationRepositoryProtocol,
            protocols.CalibrationRepositoryProtocol,
        ):
            for name, member in vars(protocol).items():
                if name.startswith("_") or not callable(member):
                    continue
                typing.get_type_hints(member)

        assert (
            typing.get_type_hints(protocols.RunRepositoryProtocol.prepare_run)["return"]
            is RunIdentity
        )
        assert (
            typing.get_type_hints(protocols.RunRepositoryProtocol.export_run)["context"]
            is RunContext
        )
        assert (
            typing.get_type_hints(protocols.ValidationRepositoryProtocol.export_run)["context"]
            is ValidationExportContext
        )
        assert (
            typing.get_type_hints(protocols.CalibrationRepositoryProtocol.load_snapshot)["return"]
            is CalibrationSnapshot
        )

        repeatability_hints = typing.get_type_hints(
            protocols.ValidationRepositoryProtocol.export_fixed_section_repeatability
        )
        assert repeatability_hints["request"] is FixedSectionRepeatabilityRequest
        assert repeatability_hints["rows"] == list[FixedSectionRepeatRow]
        assert (
            repeatability_hints["captures"]
            == Sequence[FixedSectionRepeatCapture] | None
        )

    def test_removed_compat_modules_stay_absent(self) -> None:
        assert importlib.util.find_spec("application.contracts") is None
        assert importlib.util.find_spec("application.state") is None
        assert importlib.util.find_spec("services.autoflow_service") is None
