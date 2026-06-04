"""Backward-compat re-exports for repository protocols.

Consumers should import from the canonical location: ``domain.protocols``.
"""

from domain.protocols import (  # noqa: F401
    CalibrationRepositoryProtocol,
    RunRepositoryProtocol,
    ValidationRepositoryProtocol,
)

__all__ = [
    "CalibrationRepositoryProtocol",
    "RunRepositoryProtocol",
    "ValidationRepositoryProtocol",
]
