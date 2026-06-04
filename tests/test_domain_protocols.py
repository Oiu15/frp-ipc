from __future__ import annotations

import importlib.util
import typing
import unittest
from collections.abc import Sequence

from domain import protocols
from domain.state import CalibrationSnapshot, RunContext, RunIdentity, ValidationExportContext
from domain.validation_models import (
    FixedSectionRepeatCapture,
    FixedSectionRepeatabilityRequest,
    FixedSectionRepeatRow,
)


class DomainProtocolsTest(unittest.TestCase):
    def test_protocol_annotations_resolve(self) -> None:
        for protocol in (
            protocols.RunRepositoryProtocol,
            protocols.ValidationRepositoryProtocol,
            protocols.CalibrationRepositoryProtocol,
        ):
            for name, member in vars(protocol).items():
                if name.startswith("_") or not callable(member):
                    continue
                with self.subTest(protocol=protocol.__name__, method=name):
                    typing.get_type_hints(member)

        self.assertIs(
            typing.get_type_hints(protocols.RunRepositoryProtocol.prepare_run)["return"],
            RunIdentity,
        )
        self.assertIs(
            typing.get_type_hints(protocols.RunRepositoryProtocol.export_run)["context"],
            RunContext,
        )
        self.assertIs(
            typing.get_type_hints(protocols.ValidationRepositoryProtocol.export_run)["context"],
            ValidationExportContext,
        )
        self.assertIs(
            typing.get_type_hints(protocols.CalibrationRepositoryProtocol.load_snapshot)["return"],
            CalibrationSnapshot,
        )

        repeatability_hints = typing.get_type_hints(
            protocols.ValidationRepositoryProtocol.export_fixed_section_repeatability
        )
        self.assertIs(repeatability_hints["request"], FixedSectionRepeatabilityRequest)
        self.assertEqual(repeatability_hints["rows"], list[FixedSectionRepeatRow])
        self.assertEqual(
            repeatability_hints["captures"],
            Sequence[FixedSectionRepeatCapture] | None,
        )

    def test_removed_application_compat_modules_stay_absent(self) -> None:
        self.assertIsNone(importlib.util.find_spec("application.contracts"))
        self.assertIsNone(importlib.util.find_spec("application.state"))


if __name__ == "__main__":
    unittest.main()
