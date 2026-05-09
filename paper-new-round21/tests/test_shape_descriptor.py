import unittest

from paper_new_selector.shape_descriptor import compute_shape_descriptor


class ShapeDescriptorTests(unittest.TestCase):
    def test_compute_shape_descriptor_reports_expected_statistics(self):
        descriptor = compute_shape_descriptor(
            private_lengths=[80, 100, 150, 220, 420],
            tail_threshold=350,
            short_threshold=120,
        )
        self.assertEqual(descriptor.median_len, 150.0)
        self.assertEqual(descriptor.p75_len, 220.0)
        self.assertAlmostEqual(descriptor.tail_ratio, 0.2)
        self.assertAlmostEqual(descriptor.short_ratio, 0.4)
        self.assertGreater(descriptor.iqr_len, 0.0)


if __name__ == "__main__":
    unittest.main()
