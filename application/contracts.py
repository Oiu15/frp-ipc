"""Backward-compat re-exports for types that have moved to lower layers.

Consumers should import from the canonical locations:

* ``RunRepositoryProtocol``, ``ValidationRepositoryProtocol``,
  ``CalibrationRepositoryProtocol`` — ``domain.protocols``
* ``EventSink``, ``EventPayload``, ``RawPoint`` — ``events.protocols``
* ``ValidationActionCancelled``, ``ValidationActionGateway`` —
  ``machine.validation_gateway``
"""

# Protocol re-exports (canonical location: domain.protocols)
from domain.protocols import (  # noqa: F401
    CalibrationRepositoryProtocol,
    RunRepositoryProtocol,
    ValidationRepositoryProtocol,
)

# Event re-exports (canonical location: events.protocols)
from events.protocols import EventPayload, EventSink, RawPoint  # noqa: F401, E402

# Validation re-exports (canonical location: machine.validation_gateway)
from machine.validation_gateway import ValidationActionCancelled, ValidationActionGateway  # noqa: F401, E402

__all__ = [
    "CalibrationRepositoryProtocol",
    "EventPayload",
    "EventSink",
    "RawPoint",
    "RunRepositoryProtocol",
    "ValidationActionCancelled",
    "ValidationActionGateway",
    "ValidationRepositoryProtocol",
]
