from __future__ import annotations

from pathlib import Path

import pytest

from config.addresses import (
    DEFAULT_GAUGE_PORT,
    DEFAULT_PLC_IP,
    DEFAULT_PLC_PORT,
    DEFAULT_UNIT_ID,
    FLOAT64_WORD_ORDER,
    POLL_INTERVAL_S,
)
from config.schema import GaugeConfig, PlcConfig, default_app_config


class TestConfigSchema:
    """Config schema — default values match the constants they claim to wrap."""

    # ------------------------------------------------------------------
    # PlcConfig / GaugeConfig — table of (config_cls, attr, expected)
    # ------------------------------------------------------------------

    _CONFIG_DEFAULTS: list[tuple[type, str, object]] = [
        # PlcConfig
        (PlcConfig, "ip",              DEFAULT_PLC_IP),
        (PlcConfig, "port",            DEFAULT_PLC_PORT),
        (PlcConfig, "unit_id",         DEFAULT_UNIT_ID),
        (PlcConfig, "poll_interval_s", POLL_INTERVAL_S),
        (PlcConfig, "word_order",      FLOAT64_WORD_ORDER),
        # GaugeConfig
        (GaugeConfig, "port",        DEFAULT_GAUGE_PORT),
        (GaugeConfig, "baud",        115200),
        (GaugeConfig, "timeout_s",   0.5),
        (GaugeConfig, "eol",         "\r"),
        (GaugeConfig, "request_cmd", "M1,1"),
        (GaugeConfig, "bytesize",    8),
        (GaugeConfig, "parity",      "N"),
        (GaugeConfig, "stopbits",    1),
    ]

    @pytest.mark.parametrize(("config_cls", "attr", "expected"), _CONFIG_DEFAULTS)
    def test_config_default_matches_constant(self, config_cls: type, attr: str, expected: object) -> None:
        config = config_cls()
        assert getattr(config, attr) == expected

    # ------------------------------------------------------------------
    # default_app_config — explicit root
    # ------------------------------------------------------------------

    def test_default_app_config_accepts_explicit_app_root(self) -> None:
        root = Path("C:/tmp/frp-ipc-test-root")
        config = default_app_config(root)

        assert config.paths.app_root_dir == root
        assert config.paths.recipe_profile_name == "FRP_IPC"
        assert config.paths.fallback_recipe_dir == Path("./data/recipes")
