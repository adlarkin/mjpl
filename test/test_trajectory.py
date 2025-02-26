import unittest

import mujoco
import numpy as np
from robot_descriptions.loaders.mujoco import load_robot_description

import mj_maniPlan.trajectory as traj
from mj_maniPlan.collision_ruleset import CollisionRuleset
from mj_maniPlan.joint_group import JointGroup
from mj_maniPlan.utils import random_valid_config


class TestTrajectory(unittest.TestCase):
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
        arm_jg = JointGroup(model, arm_joints)

        data.qpos = model.keyframe("home").qpos
        q_init_arm = arm_jg.qpos(data)

        rng = np.random.default_rng(seed=5)
        q_goal = random_valid_config(rng, arm_jg, data, CollisionRuleset(model))

        path = [q_init_arm, q_goal]
        limits = traj.TrajectoryLimits(
            jg=arm_jg,
            max_velocity=np.ones(dof),
            min_velocity=-np.ones(dof),
            max_acceleration=np.ones(dof),
            min_acceleration=-np.ones(dof),
            jerk=np.ones(dof),
        )
        t = traj.generate_trajectory(path, limits, model.opt.timestep)

        np.testing.assert_allclose(q_goal, t.configurations[-1], rtol=1e-5, atol=1e-8)
        # The final velocities and accelerations should be the zero vector,
        # so enforce absolute tolerance only since tolerance is not scale-dependent.
        np.testing.assert_allclose(np.zeros(dof), t.velocities[-1], rtol=0, atol=1e-8)
        np.testing.assert_allclose(
            np.zeros(dof), t.accelerations[-1], rtol=0, atol=1e-8
        )


if __name__ == "__main__":
    unittest.main()
