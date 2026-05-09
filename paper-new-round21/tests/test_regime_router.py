import unittest

from paper_new_selector.regime_router import route_budget_regime
from paper_new_selector.shape_descriptor import ShapeDescriptor


class RegimeRouterTests(unittest.TestCase):
    def test_route_budget_regime_returns_broad_tail(self):
        descriptor = ShapeDescriptor(
            median_len=210.0,
            p75_len=460.0,
            tail_ratio=0.35,
            short_ratio=0.05,
            iqr_len=260.0,
        )
        decision = route_budget_regime(
            descriptor,
            router_cfg={
                "tau_center": 0.0,
                "delta_router": 0.35,
                "screening_reference": {
                    "median_len": {"mean": 150.0, "std": 50.0},
                    "p75_len": {"mean": 335.0, "std": 90.0},
                    "iqr_len": {"mean": 180.0, "std": 60.0},
                },
            },
        )
        self.assertEqual(decision.regime, "broad_tail")
        self.assertGreater(decision.shape_score, 0.35)

    def test_route_budget_regime_returns_compact_structured(self):
        descriptor = ShapeDescriptor(
            median_len=90.0,
            p75_len=140.0,
            tail_ratio=0.02,
            short_ratio=0.50,
            iqr_len=60.0,
        )
        decision = route_budget_regime(
            descriptor,
            router_cfg={
                "tau_center": 0.0,
                "delta_router": 0.35,
                "screening_reference": {
                    "median_len": {"mean": 150.0, "std": 50.0},
                    "p75_len": {"mean": 335.0, "std": 90.0},
                    "iqr_len": {"mean": 180.0, "std": 60.0},
                },
            },
        )
        self.assertEqual(decision.regime, "compact_structured")


if __name__ == "__main__":
    unittest.main()
