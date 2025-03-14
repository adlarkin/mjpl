import unittest

import numpy as np

import mj_maniPlan.utils as utils


class TestUtils(unittest.TestCase):
    def test_fill_path(self):
        path = [
            np.array([-0.1, 0.0]),
            np.array([0.4, 0.0]),
        ]

        filled_path = utils.fill_path(path, 1, 0.2)
        self.assertEqual(len(filled_path), 3)
        np.testing.assert_equal(filled_path[0], path[0])
        np.testing.assert_allclose(
            filled_path[1], np.array([0.15, 0.0]), rtol=0, atol=1e-8
        )
        np.testing.assert_equal(filled_path[-1], path[-1])

        filled_path = utils.fill_path(path, 1, 0.75)
        self.assertEqual(len(filled_path), len(path))
        np.testing.assert_equal(filled_path[0], path[0])
        np.testing.assert_equal(filled_path[-1], path[-1])

        filled_path = utils.fill_path(path, 4, 0.1)
        self.assertEqual(len(filled_path), 6)
        np.testing.assert_equal(filled_path[0], path[0])
        np.testing.assert_allclose(filled_path[1], np.array([0.0, 0.0]))
        np.testing.assert_allclose(filled_path[2], np.array([0.1, 0.0]))
        np.testing.assert_allclose(filled_path[3], np.array([0.2, 0.0]))
        np.testing.assert_allclose(filled_path[4], np.array([0.3, 0.0]))
        np.testing.assert_equal(filled_path[-1], path[-1])

        with self.assertRaisesRegex(ValueError, "num_intermediate_points must be > 0"):
            utils.fill_path(path, 0, 1)
            utils.fill_path(path, -5, 1)


if __name__ == "__main__":
    unittest.main()
