from __future__ import annotations

import unittest
from unittest.mock import patch

from thesis_platform.core.privacy import PrivacyLedger


class PrivacyLedgerDeviceTests(unittest.TestCase):
    def test_real_dp_uses_configured_device_and_round_trips_in_report(self) -> None:
        captured: dict[str, str] = {}

        class FakePolicy:
            enabled = True
            dp_enabled = True
            epsilon = 1.0
            delta = 1e-5
            max_grad_norm = 1.0
            noise_multiplier = 1.0
            mode = "sample_critique_upload_proxy"

            def to_dp_config(self):
                return object()

            def snapshot(self):
                return {"dp_enabled": True, "mode": self.mode}

        class FakePrivatizer:
            def __init__(self, config, device="cpu"):
                captured["device"] = device
                self._state = {"device": device}

            def get_privacy_budget_status(self):
                return {
                    "query_count": 0,
                    "epsilon_spent": 0.0,
                    "budget_left": 1.0,
                    "budget_exceeded": False,
                }

            def export_state(self):
                return dict(self._state)

            def restore_state(self, state):
                self._state = dict(state)

        with patch("thesis_platform.core.privacy.DP_PRIVACY_AVAILABLE", True), patch(
            "thesis_platform.core.privacy.create_dp_config_from_dict",
            return_value=object(),
        ), patch("thesis_platform.core.privacy.DPPrivatizer", FakePrivatizer):
            ledger = PrivacyLedger(policy=FakePolicy(), device="cuda:1")
            self.assertEqual(captured["device"], "cuda:1")

            report = ledger.report()
            self.assertEqual(report["device"], "cuda:1")

            restored = PrivacyLedger.restore_from_report(report)
            self.assertEqual(restored.device, "cuda:1")


if __name__ == "__main__":
    unittest.main()
