import unittest

import numpy as np
from scipy.interpolate import make_interp_spline

from mjpl.trajectory.trajectory_interface import Trajectory
from mjpl.trajectory.utils import _add_intermediate_waypoint, _path_timing


class TestTrajectoryUtils(unittest.TestCase):
    def setUp(self):
        self.path = [
            np.array([0, 0]),
            np.array([1, 1]),
            np.array([2, 1]),
            np.array([2, 0]),
        ]
        self.splx = np.linspace(0, 1, len(self.path))
        self.spline = make_interp_spline(self.splx, self.path)

        num_trajectory_points = 100
        positions_array = self.spline(np.linspace(0, 1, num_trajectory_points))
        self.trajectory = Trajectory(
            dt=1 / num_trajectory_points,
            positions=[row for row in positions_array],
            velocities=[],
            accelerations=[],
        )

    def test_path_timing(self):
        times = _path_timing(self.path, self.trajectory)
        self.assertEqual(len(self.splx), len(times))
        # Since waypoints in a trajectory are discretized by dt, the path timing and
        # "ground truth" timing should be within dt.
        for i in range(len(self.splx)):
            self.assertLessEqual(abs(self.splx[i] - times[i]), self.trajectory.dt)
        # Path timing should be monotonically increasing.
        for i in range(len(times) - 1):
            self.assertLess(times[i], times[i + 1])

    def test_add_intermediate_waypoint(self):
        times = _path_timing(self.path, self.trajectory)

        # Using timestamps outside of the path timing range should do nothing since
        # there's no corresponding segment in the path.
        path_copy = self.path.copy()
        _add_intermediate_waypoint(path_copy, times, -1)
        self.assertListEqual(path_copy, self.path)
        _add_intermediate_waypoint(path_copy, times, 2)
        self.assertListEqual(path_copy, self.path)

        # Using a timestamp within a path segment should result in an intermediate
        # waypoint added to that segment.
        path_copy = self.path.copy()
        _add_intermediate_waypoint(
            path_copy, times, self.splx[1] + ((self.splx[2] - self.splx[1]) / 3)
        )
        self.assertEqual(len(path_copy), len(self.path) + 1)
        np.testing.assert_equal(path_copy[0], self.path[0])
        np.testing.assert_equal(path_copy[1], self.path[1])
        np.testing.assert_allclose(
            path_copy[2], (self.path[1] + self.path[2]) / 2, rtol=0, atol=1e-8
        )
        np.testing.assert_equal(path_copy[3], self.path[2])
        np.testing.assert_equal(path_copy[4], self.path[3])

        # Edge case: timestamp corresponds to the start of the path.
        path_copy = self.path.copy()
        _add_intermediate_waypoint(path_copy, times, times[0])
        self.assertEqual(len(path_copy), len(self.path) + 1)
        np.testing.assert_equal(path_copy[0], self.path[0])
        np.testing.assert_allclose(
            path_copy[1], (self.path[0] + self.path[1]) / 2, rtol=0, atol=1e-8
        )
        np.testing.assert_equal(path_copy[2], self.path[1])
        np.testing.assert_equal(path_copy[3], self.path[2])
        np.testing.assert_equal(path_copy[4], self.path[3])

        # Edge case: timestamp corresponds to the end of the path.
        path_copy = self.path.copy()
        _add_intermediate_waypoint(path_copy, times, times[-1])
        self.assertEqual(len(path_copy), len(self.path) + 1)
        np.testing.assert_equal(path_copy[0], self.path[0])
        np.testing.assert_equal(path_copy[1], self.path[1])
        np.testing.assert_equal(path_copy[2], self.path[2])
        np.testing.assert_allclose(
            path_copy[3], (self.path[2] + self.path[3]) / 2, rtol=0, atol=1e-8
        )
        np.testing.assert_equal(path_copy[4], self.path[3])

        # Edge case: timestamp corresponds to an existing path waypoint that's not the
        # start or end of the path.
        path_copy = self.path.copy()
        _add_intermediate_waypoint(path_copy, times, times[1])
        self.assertEqual(len(path_copy), len(self.path) + 1)
        np.testing.assert_equal(path_copy[0], self.path[0])
        np.testing.assert_allclose(
            path_copy[1], (self.path[0] + self.path[1]) / 2, rtol=0, atol=1e-8
        )
        np.testing.assert_equal(path_copy[2], self.path[1])
        np.testing.assert_equal(path_copy[3], self.path[2])
        np.testing.assert_equal(path_copy[4], self.path[3])


if __name__ == "__main__":
    unittest.main()
