import unittest

import numpy as np

import mjpl


class TestDrakeTrajectoryGenerator(unittest.TestCase):
    def test_generate_trajectory(self):
        dof = 7
        dt = 0.002

        joint_limits = (-np.ones(dof) * np.pi, np.ones(dof) * np.pi)
        velocity_limits = (-np.ones(dof), np.ones(dof))
        acceleration_limits = (-np.ones(dof), np.ones(dof))

        traj_generator = mjpl.DrakeTrajectoryGenerator(
            dt=dt,
            joint_limits=joint_limits,
            velocity_limits=velocity_limits,
            acceleration_limits=acceleration_limits,
        )

        rng = np.random.default_rng(seed=5)
        waypoints = [
            rng.random(dof),
            rng.random(dof),
            rng.random(dof),
            rng.random(dof),
            rng.random(dof),
        ]

        t = traj_generator.generate_trajectory(waypoints)
        self.assertIsNotNone(t)
        self.assertEqual(t.dt, dt)
        np.testing.assert_equal(t.q_init, waypoints[0])

        # Ensure limits are enforced, with some tolerance for floating point error.
        tolerance = 1e-8
        for p in t.positions:
            self.assertTrue(np.all(p >= joint_limits[0] - tolerance))
            self.assertTrue(np.all(p <= joint_limits[1] + tolerance))
        for v in t.velocities:
            self.assertTrue(np.all(v >= velocity_limits[0] - tolerance))
            self.assertTrue(np.all(v <= velocity_limits[1] + tolerance))
        for a in t.accelerations:
            self.assertTrue(np.all(a >= acceleration_limits[0] - tolerance))
            self.assertTrue(np.all(a <= acceleration_limits[1] + tolerance))

        # TODO: check to make sure waypoints are passed through in order?
        # This depends on whether hard or soft constraints are used for intermediate waypoints

        # Ensure trajectory achieves the goal state.
        np.testing.assert_allclose(waypoints[-1], t.positions[-1], rtol=1e-5, atol=1e-8)
        # The final velocity of the trajectory should be zero.
        np.testing.assert_allclose(np.zeros(dof), t.velocities[-1], rtol=0, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
