import unittest

import mujoco
import numpy as np
from robot_descriptions.loaders.mujoco import load_robot_description

import mjpl


class TestRuckigTrajectoryGenerator(unittest.TestCase):
    def test_generate_trajectory(self):
        model = load_robot_description("ur5e_mj_description")
        data = mujoco.MjData(model)

        arm_joints = [
            model.joint("shoulder_pan_joint").id,
            model.joint("shoulder_lift_joint").id,
            model.joint("elbow_joint").id,
            model.joint("wrist_1_joint").id,
            model.joint("wrist_2_joint").id,
            model.joint("wrist_3_joint").id,
        ]
        dof = len(arm_joints)
        arm_jg = mjpl.JointGroup(model, arm_joints)

        data.qpos = model.keyframe("home").qpos
        q_init_arm = arm_jg.qpos(data)

        rng = np.random.default_rng(seed=5)
        q_goal = mjpl.random_valid_config(
            rng, arm_jg, data, mjpl.CollisionRuleset(model)
        )

        traj_generator = mjpl.RuckigTrajectoryGenerator(
            dt=model.opt.timestep,
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

        path = [q_init_arm, q_goal]
        t = traj_generator.generate_trajectory(path)

        # Ensure limits are enforced.
        for v in t.velocities:
            self.assertTrue(
                np.all(
                    (v >= traj_generator.min_velocity)
                    & (v <= traj_generator.max_velocity)
                )
            )
        for a in t.accelerations:
            self.assertTrue(
                np.all(
                    (a >= traj_generator.min_acceleration)
                    & (a <= traj_generator.max_acceleration)
                )
            )

        # Ensure trajectory achieves the goal state.
        np.testing.assert_allclose(q_goal, t.positions[-1], rtol=1e-5, atol=1e-8)
        # The final velocities and accelerations should be the zero vector,
        # so enforce absolute tolerance only since tolerance is not scale-dependent.
        np.testing.assert_allclose(np.zeros(dof), t.velocities[-1], rtol=0, atol=1e-8)
        np.testing.assert_allclose(
            np.zeros(dof), t.accelerations[-1], rtol=0, atol=1e-8
        )


if __name__ == "__main__":
    unittest.main()
