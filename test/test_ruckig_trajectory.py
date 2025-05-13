import unittest

import numpy as np

import mjpl


class TestRuckigTrajectoryGenerator(unittest.TestCase):
    def test_generate_trajectory(self):
        dof = 7
        dt = 0.002

        traj_generator = mjpl.RuckigTrajectoryGenerator(
            dt=dt,
            max_velocity=np.ones(dof),
            max_acceleration=np.ones(dof),
            max_jerk=np.ones(dof),
        )
        np.testing.assert_equal(
            traj_generator.min_velocity, -traj_generator.max_velocity
        )
        np.testing.assert_equal(
            traj_generator.min_acceleration, -traj_generator.max_acceleration
        )

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
            self.assertTrue(np.all(v >= traj_generator.min_velocity - tolerance))
            self.assertTrue(np.all(v <= traj_generator.max_velocity + tolerance))
        for a in t.accelerations:
            self.assertTrue(np.all(a >= traj_generator.min_acceleration - tolerance))
            self.assertTrue(np.all(a <= traj_generator.max_acceleration + tolerance))
        for i in range(len(t.accelerations)):
            prev_acc = np.zeros(dof) if i == 0 else t.accelerations[i - 1]
            curr_acc = t.accelerations[i]
            jerk = np.abs((curr_acc - prev_acc) / t.dt)
            self.assertTrue(np.all(jerk <= traj_generator.max_jerk + tolerance))

        # Ensure trajectory achieves the goal state.
        np.testing.assert_allclose(waypoints[-1], t.positions[-1], rtol=1e-5, atol=1e-8)
        # The final velocity of the trajectory should be zero.
        np.testing.assert_allclose(np.zeros(dof), t.velocities[-1], rtol=0, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
