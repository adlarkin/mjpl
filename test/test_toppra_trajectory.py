import unittest

import numpy as np

import mjpl


class TestToppraTrajectoryGenerator(unittest.TestCase):
    def test_generate_trajectory(self):
        dof = 7
        dt = 0.002

        traj_generator = mjpl.ToppraTrajectoryGenerator(
            dt=dt,
            max_velocity=np.ones(dof),
            max_acceleration=np.ones(dof),
        )

        vel_limit_min = traj_generator.velocity_constraint.vlim[:, 0]
        vel_limit_max = traj_generator.velocity_constraint.vlim[:, 1]
        np.testing.assert_equal(vel_limit_min, -vel_limit_max)

        acc_limit_min = traj_generator.acceleration_constraint.alim[:, 0]
        acc_limit_max = traj_generator.acceleration_constraint.alim[:, 1]
        np.testing.assert_equal(acc_limit_min, -acc_limit_max)

        rng = np.random.default_rng(seed=5)
        waypoints = [
            rng.random(dof),
            rng.random(dof),
        ]

        t = traj_generator.generate_trajectory(waypoints)
        self.assertIsNotNone(t)
        self.assertEqual(t.dt, dt)
        np.testing.assert_equal(t.q_init, waypoints[0])

        # Ensure limits are enforced, with some tolerance for floating point error.
        tolerance = 1e-8
        for v in t.velocities:
            self.assertTrue(np.all(v >= vel_limit_min - tolerance))
            self.assertTrue(np.all(v <= vel_limit_max + tolerance))
        for a in t.accelerations:
            self.assertTrue(np.all(a >= acc_limit_min - tolerance))
            self.assertTrue(np.all(a <= acc_limit_max + tolerance))

        # Ensure trajectory achieves the goal state.
        np.testing.assert_allclose(waypoints[-1], t.positions[-1], rtol=1e-5, atol=1e-8)
        # The final velocity of the trajectory should be zero.
        np.testing.assert_allclose(np.zeros(dof), t.velocities[-1], rtol=0, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
