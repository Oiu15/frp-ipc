from __future__ import annotations

import typing
import unittest

import application.contracts as contracts
from application.state import CalibrationSnapshot, RunContext, RunIdentity, ValidationExportContext


class ApplicationContractsTest(unittest.TestCase):
    def test_state_models_are_not_reexported(self) -> None:
        for name in ("CalibrationSnapshot", "RunContext", "RunIdentity", "ValidationExportContext"):
            self.assertNotIn(name, contracts.__all__)
            self.assertFalse(hasattr(contracts, name))

    def test_protocol_state_annotations_still_resolve(self) -> None:
        self.assertIs(
            typing.get_type_hints(contracts.RunRepositoryProtocol.prepare_run)["return"],
            RunIdentity,
        )
        self.assertIs(
            typing.get_type_hints(contracts.RunRepositoryProtocol.export_run)["context"],
            RunContext,
        )
        self.assertIs(
            typing.get_type_hints(contracts.ValidationRepositoryProtocol.export_run)["context"],
            ValidationExportContext,
        )
        self.assertIs(
            typing.get_type_hints(contracts.CalibrationRepositoryProtocol.load_snapshot)["return"],
            CalibrationSnapshot,
        )


if __name__ == "__main__":
    unittest.main()
