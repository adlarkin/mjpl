import pdb
import unittest

import mujoco
import mujoco.viewer
import numpy as np
from robot_descriptions.loaders.mujoco import load_robot_description

import mjpl


class TestPoseConstraint(unittest.TestCase):
    def test_constraint(self):
        model = load_robot_description("ur5e_mj_description")
        data = mujoco.MjData(model)
        site_name = "attachment_site"

        q_init = model.keyframe("home").qpos
        data.qpos = q_init
        mujoco.mj_kinematics(model, data)
        ee_init_pose = mjpl.site_pose(data, site_name)

        q_rand = mjpl.random_config(
            model,
            q_init,
            mjpl.all_joints(model),
            seed=123,
            constraints=[mjpl.CollisionConstraint(model)],
        )
        """
        data.qpos = q_rand
        mujoco.mj_kinematics(model, data)
        ee_rand_pose = mjpl.site_pose(data, site_name)
        """

        pose_constraint = mjpl.PoseConstraint(
            model,
            (-np.inf, np.inf),
            (-np.inf, np.inf),
            (-np.inf, np.inf),
            (-0.1, 0.1),
            (-0.1, 0.1),
            (-np.inf, np.inf),
            ee_init_pose,
            site_name,
            tolerance=0.001,
            q_step=np.inf,
        )

        # TODO: replace this with actual test logic, this is currently for debugging
        with mujoco.viewer.launch_passive(
            model=model, data=data, show_right_ui=False
        ) as viewer:
            # show initial config
            data.qpos = q_init
            mujoco.mj_kinematics(model, data)
            viewer.sync()
            pdb.set_trace()

            # show random config
            data.qpos = q_rand
            mujoco.mj_kinematics(model, data)
            viewer.sync()
            pdb.set_trace()

            # show constrained config
            q_constrained = pose_constraint.apply(q_init, q_rand)
            if q_constrained is None:
                print("unable to constrain config")
                return
            data.qpos = q_constrained
            mujoco.mj_kinematics(model, data)
            viewer.sync()
            pdb.set_trace()


if __name__ == "__main__":
    unittest.main()
