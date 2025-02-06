'''
Example of how to run RRT on a franka panda and visualize the paths
(with and without shortcutting).
'''

import mujoco
import mujoco.viewer
import numpy as np
import time

from mj_maniPlan.rrt import (
    RRT,
    RRTOptions,
)
import example_utils as ex_utils
import mj_maniPlan.utils as utils
import mj_maniPlan.visualization as viz


if __name__ == '__main__':
    model = ex_utils.load_panda_model()
    data = mujoco.MjData(model)

    # The joints to sample during planning.
    # Since this example executes planning for the arm,
    # the finger joints of the gripper are excluded.
    joint_names = ex_utils.panda_arm_joints()

    # Random number generator that's used for sampling joint configurations.
    rng_seed = None

    # Generate valid initial/goal configurations.
    joint_qpos_addrs = utils.joint_names_to_qpos_addrs(joint_names, model)
    lower_limits, upper_limits = utils.joint_limits(joint_names, model)
    rng = np.random.default_rng(seed=rng_seed)
    q_init = utils.random_valid_config(rng, lower_limits, upper_limits, joint_qpos_addrs, model, data)
    q_goal = utils.random_valid_config(rng, lower_limits, upper_limits, joint_qpos_addrs, model, data)

    # Set up the planner.
    epsilon = 0.05
    planner_options = RRTOptions(
        joint_names=joint_names,
        max_planning_time=10,
        epsilon=epsilon,
        shortcut_filler_epsilon=10*epsilon,
        seed=rng_seed,
        goal_biasing_probability=0.1,
    )
    planner = RRT(planner_options, model)

    print("Planning...")
    start = time.time()
    path = planner.plan(q_init, q_goal)
    if not path:
        print("Planning failed")
        exit()
    print(f"Planning took {time.time() - start}s")

    print("Shortcutting...")
    start = time.time()
    shortcut_path = planner.shortcut(path, num_attempts=len(path))
    print(f"Shortcutting took {time.time() - start}s")

    # Smooth the path by performing naive joint-space B-spline interpolation.
    # This will be used for visualization.
    spline = ex_utils.fit_path_to_spline(path)
    spline_shortcut = ex_utils.fit_path_to_spline(shortcut_path)

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        # Update the viewer's orientation to capture the scene.
        viewer.cam.lookat = [0, 0, 0.35]
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -25

        # Show the initial EE pose
        utils.fk(q_init, joint_qpos_addrs, model, data)
        init_pos, init_rot, = utils.site_pose(ex_utils._PANDA_EE_SITE, data)
        viz.add_frame(viewer.user_scn, init_pos, init_rot)

        # Show the target EE pose
        utils.fk(q_goal, joint_qpos_addrs, model, data)
        target_pos, target_rot = utils.site_pose(ex_utils._PANDA_EE_SITE, data)
        viz.add_frame(viewer.user_scn, target_pos, target_rot)

        # Show the initial and shortcut paths.
        print("Regular path is in red, shortcut path is in green")
        path_to_color = {
            spline:          [1.0, 0.0, 0.0, 1.0],
            spline_shortcut: [0.0, 1.0, 0.0, 1.0],
        }
        horizon = np.linspace(0, 1, 1000)
        for p, rgba in path_to_color.items():
            for t in horizon:
                q_t = p(t)
                utils.fk(q_t, joint_qpos_addrs, model, data)
                ee_world_pos = data.site(ex_utils._PANDA_EE_SITE).xpos
                viz.add_sphere(viewer.user_scn, ee_world_pos, 0.004, rgba)

        # Ensure the robot is at q_init and then update the viewer to show
        # the frames, paths, and initial state.
        utils.fk(q_init, joint_qpos_addrs, model, data)
        viewer.sync()

        # Keep the visualizer open until the user closes it.
        # This loop does not need to run at a high rate (doing 10hz here).
        time_between_checks = 1 / 10
        while viewer.is_running():
            time.sleep(time_between_checks)
