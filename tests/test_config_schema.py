import unittest
from pathlib import Path

from config.addresses import (
    DEFAULT_GAUGE_PORT,
    DEFAULT_PLC_IP,
    DEFAULT_PLC_PORT,
    DEFAULT_UNIT_ID,
    FLOAT64_WORD_ORDER,
    POLL_INTERVAL_S,
)
from config.schema import GaugeConfig, PlcConfig, default_app_config


class ConfigSchemaTest(unittest.TestCase):
    def test_plc_config_defaults_match_address_constants(self) -> None:
        config = PlcConfig()

        self.assertEqual(config.ip, DEFAULT_PLC_IP)
        self.assertEqual(config.port, DEFAULT_PLC_PORT)
        self.assertEqual(config.unit_id, DEFAULT_UNIT_ID)
        self.assertEqual(config.poll_interval_s, POLL_INTERVAL_S)
        self.assertEqual(config.word_order, FLOAT64_WORD_ORDER)

    def test_gauge_config_defaults_match_current_worker_defaults(self) -> None:
        config = GaugeConfig()

        self.assertEqual(config.port, DEFAULT_GAUGE_PORT)
        self.assertEqual(config.baud, 115200)
        self.assertEqual(config.timeout_s, 0.5)
        self.assertEqual(config.eol, "\r")
        self.assertEqual(config.request_cmd, "M1,1")
        self.assertEqual(config.bytesize, 8)
        self.assertEqual(config.parity, "N")
        self.assertEqual(config.stopbits, 1)

    def test_default_app_config_accepts_explicit_app_root(self) -> None:
        root = Path("C:/tmp/frp-ipc-test-root")

        config = default_app_config(root)

        self.assertEqual(config.paths.app_root_dir, root)
        self.assertEqual(config.paths.recipe_profile_name, "FRP_IPC")
        self.assertEqual(config.paths.fallback_recipe_dir, Path("./data/recipes"))


if __name__ == "__main__":
    unittest.main()
