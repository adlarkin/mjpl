import unittest

import numpy as np
from scipy.interpolate import make_interp_spline

from mjpl.trajectory.trajectory_interface import Trajectory
from mjpl.trajectory.utils import _add_intermediate_waypoint, _waypoint_timing


class TestTrajectoryUtils(unittest.TestCase):
    def setUp(self):
        self.waypoints = [
            np.array([0, 0]),
            np.array([1, 1]),
            np.array([2, 1]),
            np.array([2, 0]),
        ]
        self.splx = np.linspace(0, 1, len(self.waypoints))
        self.spline = make_interp_spline(self.splx, self.waypoints)

        num_trajectory_points = 100
        positions_array = self.spline(np.linspace(0, 1, num_trajectory_points))
        self.trajectory = Trajectory(
            dt=1 / num_trajectory_points,
            q_init=positions_array[0],
            positions=[row for row in positions_array],
            velocities=[],
            accelerations=[],
        )

    def test_waypoint_timing(self):
        times = _waypoint_timing(self.waypoints, self.trajectory)
        self.assertEqual(len(self.splx), len(times))
        # Since waypoints in a trajectory are discretized by dt, the waypoint timing
        # and "ground truth" timing should be within dt.
        for i in range(len(self.splx)):
            self.assertLessEqual(abs(self.splx[i] - times[i]), self.trajectory.dt)
        # Waypoint timing should be monotonically increasing.
        for i in range(len(times) - 1):
            self.assertLess(times[i], times[i + 1])

    def test_add_intermediate_waypoint(self):
        times = _waypoint_timing(self.waypoints, self.trajectory)

        # Using timestamps outside of the waypoint timing range should do nothing since
        # there's no corresponding segment in the waypoint list.
        waypoints_copy = self.waypoints.copy()
        self.assertFalse(_add_intermediate_waypoint(waypoints_copy, times, -1))
        self.assertListEqual(waypoints_copy, self.waypoints)
        self.assertFalse(_add_intermediate_waypoint(waypoints_copy, times, 2))
        self.assertListEqual(waypoints_copy, self.waypoints)

        # Using a timestamp within a waypoint segment should result in an intermediate
        # waypoint added to that segment.
        waypoint_copy = self.waypoints.copy()
        self.assertTrue(
            _add_intermediate_waypoint(
                waypoint_copy, times, self.splx[1] + ((self.splx[2] - self.splx[1]) / 3)
            )
        )
        self.assertEqual(len(waypoint_copy), len(self.waypoints) + 1)
        np.testing.assert_equal(waypoint_copy[0], self.waypoints[0])
        np.testing.assert_equal(waypoint_copy[1], self.waypoints[1])
        np.testing.assert_allclose(
            waypoint_copy[2],
            (self.waypoints[1] + self.waypoints[2]) / 2,
            rtol=0,
            atol=1e-8,
        )
        np.testing.assert_equal(waypoint_copy[3], self.waypoints[2])
        np.testing.assert_equal(waypoint_copy[4], self.waypoints[3])

        # Edge case: timestamp corresponds to the start of the waypoints.
        waypoint_copy = self.waypoints.copy()
        self.assertTrue(_add_intermediate_waypoint(waypoint_copy, times, times[0]))
        self.assertEqual(len(waypoint_copy), len(self.waypoints) + 1)
        np.testing.assert_equal(waypoint_copy[0], self.waypoints[0])
        np.testing.assert_allclose(
            waypoint_copy[1],
            (self.waypoints[0] + self.waypoints[1]) / 2,
            rtol=0,
            atol=1e-8,
        )
        np.testing.assert_equal(waypoint_copy[2], self.waypoints[1])
        np.testing.assert_equal(waypoint_copy[3], self.waypoints[2])
        np.testing.assert_equal(waypoint_copy[4], self.waypoints[3])

        # Edge case: timestamp corresponds to the end of the waypoints.
        waypoint_copy = self.waypoints.copy()
        self.assertTrue(_add_intermediate_waypoint(waypoint_copy, times, times[-1]))
        self.assertEqual(len(waypoint_copy), len(self.waypoints) + 1)
        np.testing.assert_equal(waypoint_copy[0], self.waypoints[0])
        np.testing.assert_equal(waypoint_copy[1], self.waypoints[1])
        np.testing.assert_equal(waypoint_copy[2], self.waypoints[2])
        np.testing.assert_allclose(
            waypoint_copy[3],
            (self.waypoints[2] + self.waypoints[3]) / 2,
            rtol=0,
            atol=1e-8,
        )
        np.testing.assert_equal(waypoint_copy[4], self.waypoints[3])

        # Edge case: timestamp corresponds to an existing waypoint that's not the
        # start or end of the waypoint list.
        waypoint_copy = self.waypoints.copy()
        self.assertTrue(_add_intermediate_waypoint(waypoint_copy, times, times[1]))
        self.assertEqual(len(waypoint_copy), len(self.waypoints) + 1)
        np.testing.assert_equal(waypoint_copy[0], self.waypoints[0])
        np.testing.assert_allclose(
            waypoint_copy[1],
            (self.waypoints[0] + self.waypoints[1]) / 2,
            rtol=0,
            atol=1e-8,
        )
        np.testing.assert_equal(waypoint_copy[2], self.waypoints[1])
        np.testing.assert_equal(waypoint_copy[3], self.waypoints[2])
        np.testing.assert_equal(waypoint_copy[4], self.waypoints[3])

        with self.assertRaisesRegex(ValueError, "must be the same length"):
            _add_intermediate_waypoint([], [0.0], 0.0)


if __name__ == "__main__":
    unittest.main()
