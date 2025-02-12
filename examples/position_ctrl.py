"""
Plan a path with RRT, generate a trajectory from the path, and then execute the trajectory with position control.
The position control is done with MuJoCo's position actuators.
"""

import time

import example_utils as ex_utils
import mujoco
import mujoco.viewer
import numpy as np
import panda_utils
import ruckig_trajectory as r_traj

if __name__ == "__main__":
    visualize, use_obstacles, seed = panda_utils.parse_panda_args(
        "Generate and follow a trajectory from a RRT path with position control."
    )
    _, shortcut_path, config = panda_utils.rrt_panda(use_obstacles, seed)

    print("Generating a trajectory on the shortcut path...")
    start = time.time()
    traj = r_traj.generate_trajectory(
        len(shortcut_path[0]), config.model.opt.timestep, shortcut_path
    )
    print(f"Trajectory generation took {(time.time() - start):.4f}s")

    if visualize:
        data = mujoco.MjData(config.model)
        q_init, q_goal = shortcut_path[0], shortcut_path[-1]
        with mujoco.viewer.launch_passive(
            model=config.model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            # Update the viewer's orientation to capture the scene.
            viewer.cam.lookat = [0, 0, 0.35]
            viewer.cam.distance = 2.5
            viewer.cam.azimuth = 145
            viewer.cam.elevation = -25

            # Draw the initial and target EE pose.
            ex_utils.add_site_frame(
                viewer.user_scn,
                panda_utils._PANDA_EE_SITE,
                q_init,
                config,
            )
            ex_utils.add_site_frame(
                viewer.user_scn,
                panda_utils._PANDA_EE_SITE,
                q_goal,
                config,
            )

            # Show the shortcut path.
            ex_utils.add_path(
                viewer.user_scn,
                panda_utils._PANDA_EE_SITE,
                shortcut_path,
                config,
                [0.2, 0.6, 0.2, 0.2],
            )

            # Ensure the robot is at q_init and then update the viewer to show
            # the frames, paths, and initial state.
            config.fk(q_init, data)
            viewer.sync()

            # Indices for actuators in data.ctrl
            actuator_ids = [
                config.model.actuator(name).id
                for name in panda_utils.panda_arm_actuators()
            ]

            # Command the robot along the path via position control.
            while viewer.is_running():
                time.sleep(0.5)

                # Reset to initial state of trajectory
                mujoco.mj_resetData(config.model, data)
                config.fk(np.array(traj.configurations[0]), data)
                config.set_qvel(np.array(traj.velocities[0]), data)
                mujoco.mj_forward(config.model, data)

                for target_q in traj.configurations:
                    start_time = time.time()

                    data.ctrl[actuator_ids] = target_q
                    mujoco.mj_step(config.model, data)

                    duration = time.time() - start_time
                    if duration < traj.dt:
                        time.sleep(traj.dt - duration)
                    viewer.sync()
