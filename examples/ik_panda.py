import mink
import mujoco
import mujoco.viewer
import numpy as np
import os
import time

from mj_maniPlan.sampling import HaltonSampler
import mj_maniPlan.utils as utils

if __name__ == "__main__":
    dir = os.path.dirname(os.path.realpath(__file__))
    model_xml_path = dir + "/../models/franka_emika_panda/scene.xml"
    model = mujoco.MjModel.from_xml_path(model_xml_path)
    data = mujoco.MjData(model)

    configuration = mink.Configuration(model)

    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="ee_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
    ]

    # TODO: load scene_with_obstacles above and then add obstacle avoidance here
    hand_geoms = mink.get_body_geom_ids(model, model.body("hand").id)
    collision_pairs = [
        (hand_geoms, ["floor"]),
    ]
    limits = [
        mink.ConfigurationLimit(model=model),
        mink.CollisionAvoidanceLimit(model=model, geom_pairs=collision_pairs),
    ]

    joint_names = [
        'joint1',
        'joint2',
        'joint3',
        'joint4',
        'joint5',
        'joint6',
        'joint7',
    ]
    max_velocities = { j: np.pi for j in joint_names }
    velocity_limit = mink.VelocityLimit(model, max_velocities)
    limits.append(velocity_limit)

    rng = HaltonSampler(len(max_velocities), seed=None)
    joint_qpos_addrs = utils.joint_names_to_qpos_addrs(joint_names, model)
    lower_limits, upper_limits = utils.joint_limits(joint_names, model)
    q_init = utils.random_valid_config(rng, lower_limits, upper_limits, joint_qpos_addrs, model, data)
    q_goal = utils.random_valid_config(rng, lower_limits, upper_limits, joint_qpos_addrs, model, data)

    # extract the EE pose from q_goal
    data.qpos[joint_qpos_addrs] = q_goal
    configuration.update(data.qpos)
    ee_target_pose = configuration.get_transform_frame_to_world(
        frame_name="ee_site",
        frame_type="site",
    )

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        # Update the viewer's orientation to capture the arm movement.
        viewer.cam.lookat = [0, 0, 0.35]
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -25

        def set_and_visualize_joint_config(q: np.ndarray):
            data.qpos[joint_qpos_addrs] = q
            mujoco.mj_kinematics(model, data)
            viewer.sync()

        # Visualize kinematic updates at 60hz.
        viz_time_per_frame = 1 / 60

        while viewer.is_running():
            # Show the start configuration.
            set_and_visualize_joint_config(q_init)
            time.sleep(1)

            # set the initial state of the IK problem to q_init
            data.qpos[joint_qpos_addrs] = q_init
            configuration.update(data.qpos)

            # move the robot from q_init to q_goal by solving IK
            end_effector_task.set_target(ee_target_pose)
            pos_threshold = 1e-4
            ori_threshold = 1e-4
            pos_achieved = False
            ori_achieved = False
            while not pos_achieved or not ori_achieved:
                start_time = time.time()
                vel = mink.solve_ik(
                    configuration, tasks, model.opt.timestep, solver="quadprog", damping=1e-3, limits=limits
                )
                configuration.integrate_inplace(vel, model.opt.timestep)
                err = end_effector_task.compute_error(configuration)
                pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
                ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
                set_and_visualize_joint_config(configuration.q[joint_qpos_addrs])
                elapsed_time = time.time() - start_time
                if elapsed_time < viz_time_per_frame:
                    time.sleep(viz_time_per_frame - elapsed_time)

            time.sleep(0.25)
