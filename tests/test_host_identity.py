from __future__ import annotations

import unittest

from application._host_identity import HostIdentityMixin
from domain.state import RunSession


class _HostIdentity(HostIdentityMixin):
    def __init__(self) -> None:
        self._run_session = RunSession()


class HostIdentityMixinTest(unittest.TestCase):
    def test_run_identity_properties_delegate_to_session(self) -> None:
        host = _HostIdentity()

        host._run_serial = "serial-001"
        host._run_id = "run-001"
        host._run_start_ts = 10.5
        host._run_end_ts = 20.5

        self.assertEqual(host._run_session.serial, "serial-001")
        self.assertEqual(host._run_session.run_id, "run-001")
        self.assertEqual(host._run_session.start_ts, 10.5)
        self.assertEqual(host._run_session.end_ts, 20.5)
        self.assertEqual(host._run_serial, "serial-001")
        self.assertEqual(host._run_id, "run-001")
        self.assertEqual(host._run_start_ts, 10.5)
        self.assertEqual(host._run_end_ts, 20.5)


if __name__ == "__main__":
    unittest.main()
