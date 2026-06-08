from __future__ import annotations

VERSION = "0.6.4-alpha.1"
VERSION_TAG = f"v{VERSION}"
SOFTWARE_VERSION = VERSION_TAG

BUILD_COMMIT = ""
BUILD_TIME = ""
BUILD_CHANNEL = "dev"

__all__ = [
    "BUILD_CHANNEL",
    "BUILD_COMMIT",
    "BUILD_TIME",
    "SOFTWARE_VERSION",
    "VERSION",
    "VERSION_TAG",
]
