"""
Example of how to run RRT on a franka panda and visualize the paths
(with and without shortcutting).
"""

import time

import example_utils as ex_utils
import mujoco
import mujoco.viewer
import panda_utils

if __name__ == "__main__":
    visualize, use_obstacles, seed = panda_utils.parse_panda_args(
        "Run RRT on a franka panda model and visualize the resulting paths."
    )
    path, shortcut_path, config = panda_utils.rrt_panda(use_obstacles, seed)
    if visualize:
        data = mujoco.MjData(config.model)
        q_init, q_goal = path[0], path[-1]
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

            # Draw the initial and shortcut paths.
            print("Regular path is in red, shortcut path is in green")
            ex_utils.add_path(
                viewer.user_scn,
                panda_utils._PANDA_EE_SITE,
                path,
                config,
                [1.0, 0.0, 0.0, 1.0],
            )
            ex_utils.add_path(
                viewer.user_scn,
                panda_utils._PANDA_EE_SITE,
                shortcut_path,
                config,
                [0.0, 1.0, 0.0, 1.0],
            )

            # Ensure the robot is at q_init and then update the viewer to show
            # the frames, paths, and initial state.
            config.fk(q_init, data)
            viewer.sync()

            # Keep the visualizer open until the user closes it.
            # This loop does not need to run at a high rate (doing 10hz here).
            time_between_checks = 1 / 10
            while viewer.is_running():
                time.sleep(time_between_checks)
