"""
Plan a path with RRT, generate a trajectory from the path, and then execute the trajectory with computed torque control.
"""

import time

import example_utils as ex_utils
import mujoco
import mujoco.viewer
import numpy as np
import panda_utils
import ruckig_trajectory as r_traj

from mj_maniPlan.configuration import Configuration


# Compute the list of joint torques that need to be applied to follow a trajectory.
# These torques should be set via data.qfrc_applied
def computed_torque_ctrl(trajectory: r_traj.Trajectory, config: Configuration):
    data = mujoco.MjData(config.model)
    config.fk(np.array(trajectory.configurations[0]), data)
    config.set_qvel(np.array(trajectory.velocities[0]), data)
    mujoco.mj_forward(config.model, data)

    ctrl_torques = []
    for t in range(len(traj.configurations)):
        # computed torque control with PD feedback for position/velocity error
        error = traj.configurations[t] - config.qpos(data)
        error_dot = traj.velocities[t] - config.qvel(data)
        # NOTE: these gains may need to be tuned
        Kp = np.eye(len(config.joint_names)) * 1.0
        Kd = np.eye(len(config.joint_names)) * 1.0
        data.qacc[:] = 0
        target_acc = traj.accelerations[t] + Kd.dot(error_dot) + Kp.dot(error)
        config.set_qacc(target_acc, data)
        mujoco.mj_inverse(config.model, data)
        ctrl_tau = data.qfrc_inverse[config.qpos_addrs]
        data.qfrc_applied[config.qpos_addrs] = ctrl_tau
        mujoco.mj_step(config.model, data)

        ctrl_torques.append(ctrl_tau)
    return ctrl_torques


if __name__ == "__main__":
    visualize, use_obstacles, seed = panda_utils.parse_panda_args(
        "Generate and follow a trajectory from a RRT path with computed torque control."
    )
    _, shortcut_path, config = panda_utils.rrt_panda(use_obstacles, seed)

    print("Generating a trajectory on the shortcut path...")
    start = time.time()
    traj = r_traj.generate_trajectory(
        len(shortcut_path[0]), config.model.opt.timestep, shortcut_path
    )
    print(f"Trajectory generation took {(time.time() - start):.4f}s")

    # Since we are doing computed torque control, we need to "disable" position actuators
    # so that data.ctrl is not part of the applied force when we step simulation forward.
    config.model.actuator_gainprm[:] = 0
    config.model.actuator_biasprm[:] = 0
    ctrl_torques = computed_torque_ctrl(traj, config)

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

            # Command the robot along the path via computed torque control.
            while viewer.is_running():
                time.sleep(0.5)

                # Reset to initial state of trajectory
                mujoco.mj_resetData(config.model, data)
                config.fk(np.array(traj.configurations[0]), data)
                config.set_qvel(np.array(traj.velocities[0]), data)
                mujoco.mj_forward(config.model, data)

                for u_t in ctrl_torques:
                    start_time = time.time()

                    data.qfrc_applied[config.qpos_addrs] = u_t
                    mujoco.mj_step(config.model, data)

                    duration = time.time() - start_time
                    if duration < traj.dt:
                        time.sleep(traj.dt - duration)
                    viewer.sync()
