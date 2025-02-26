import unittest

import mujoco
import numpy as np
from robot_descriptions.loaders.mujoco import load_robot_description

from mj_maniPlan.joint_group import JointGroup, joint_ids_to_qpos_addrs, joint_limits


def load_panda() -> tuple[mujoco.MjModel, list[int], list[int]]:
    model = load_robot_description("panda_mj_description")
    arm_joint_ids = [
        model.joint("joint1").id,
        model.joint("joint2").id,
        model.joint("joint3").id,
        model.joint("joint4").id,
        model.joint("joint5").id,
        model.joint("joint6").id,
        model.joint("joint7").id,
    ]
    hand_joint_ids = [
        model.joint("finger_joint1").id,
        model.joint("finger_joint2").id,
    ]
    return model, arm_joint_ids, hand_joint_ids


class TestJointGroupHelpers(unittest.TestCase):
    def setUp(self):
        self.model, self.arm_joint_ids, self.hand_joint_ids = load_panda()

    def test_joint_ids_to_qpos_addrs(self):
        arm_addrs = joint_ids_to_qpos_addrs(self.model, self.arm_joint_ids)
        self.assertEqual(len(self.arm_joint_ids), arm_addrs.size)
        for i in range(len(self.arm_joint_ids)):
            qpos_addr = self.model.joint(self.arm_joint_ids[i]).qposadr
            # joint_ids_to_qpos_addrs assumes 1 DOF joints
            self.assertEqual(1, qpos_addr.size)
            qpos_addr = qpos_addr.item()
            self.assertEqual(arm_addrs[i], qpos_addr)

        hand_addrs = joint_ids_to_qpos_addrs(self.model, self.hand_joint_ids)
        self.assertEqual(len(self.hand_joint_ids), hand_addrs.size)
        for i in range(len(self.hand_joint_ids)):
            qpos_addr = self.model.joint(self.hand_joint_ids[i]).qposadr
            # joint_ids_to_qpos_addrs assumes 1 DOF joints
            self.assertEqual(1, qpos_addr.size)
            qpos_addr = qpos_addr.item()
            self.assertEqual(hand_addrs[i], qpos_addr)

    def test_joint_limits(self):
        arm_lower, arm_upper = joint_limits(self.model, self.arm_joint_ids)
        self.assertEqual(len(self.arm_joint_ids), arm_lower.size)
        self.assertEqual(len(self.arm_joint_ids), arm_upper.size)
        for i in range(len(self.arm_joint_ids)):
            joint = self.model.joint(self.arm_joint_ids[i])
            self.assertEqual(arm_lower[i], joint.range[0])
            self.assertEqual(arm_upper[i], joint.range[1])

        hand_lower, hand_upper = joint_limits(self.model, self.hand_joint_ids)
        self.assertEqual(len(self.hand_joint_ids), hand_lower.size)
        self.assertEqual(len(self.hand_joint_ids), hand_upper.size)
        for i in range(len(self.hand_joint_ids)):
            joint = self.model.joint(self.hand_joint_ids[i])
            self.assertEqual(hand_lower[i], joint.range[0])
            self.assertEqual(hand_upper[i], joint.range[1])


class TestJointGroup(unittest.TestCase):
    def setUp(self):
        self.model, self.arm_joint_ids, self.hand_joint_ids = load_panda()
        self.data = mujoco.MjData(self.model)

    def test_random_config(self):
        jg = JointGroup(self.model, self.arm_joint_ids)
        q_rand = jg.random_config(np.random.default_rng(seed=5))
        self.assertEqual(len(self.arm_joint_ids), q_rand.size)
        lower_limits, upper_limits = joint_limits(self.model, self.arm_joint_ids)
        for i in range(len(self.arm_joint_ids)):
            self.assertGreaterEqual(q_rand[i], lower_limits[i])
            self.assertLessEqual(q_rand[i], upper_limits[i])

    def test_fk(self):
        jg = JointGroup(self.model, self.arm_joint_ids)

        ref_data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(
            self.model, ref_data, self.model.keyframe("home").id
        )
        q_arm = jg.qpos(ref_data)

        self.assertFalse(np.array_equal(q_arm, jg.qpos(self.data)))
        jg.fk(q_arm, self.data)
        self.assertTrue(np.array_equal(q_arm, jg.qpos(self.data)))

    def test_qpos(self):
        arm_addrs = joint_ids_to_qpos_addrs(self.model, self.arm_joint_ids)
        hand_addrs = joint_ids_to_qpos_addrs(self.model, self.hand_joint_ids)
        q_robot = self.model.keyframe("home").qpos
        q_arm = q_robot[arm_addrs]
        q_hand = q_robot[hand_addrs]

        mujoco.mj_resetDataKeyframe(
            self.model, self.data, self.model.keyframe("home").id
        )
        arm_jg = JointGroup(self.model, self.arm_joint_ids)
        self.assertTrue(np.array_equal(q_arm, arm_jg.qpos(self.data)))
        hand_jg = JointGroup(self.model, self.hand_joint_ids)
        self.assertTrue(np.array_equal(q_hand, hand_jg.qpos(self.data)))

    def test_joint_ids(self):
        arm_jg = JointGroup(self.model, self.arm_joint_ids)
        self.assertListEqual(self.arm_joint_ids, arm_jg.joint_ids)
        hand_jg = JointGroup(self.model, self.hand_joint_ids)
        self.assertListEqual(self.hand_joint_ids, hand_jg.joint_ids)


if __name__ == "__main__":
    unittest.main()
