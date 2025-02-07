"""
Plan a path with RRT, generate a trajectory from the path, and then execute the trajectory with computed torque control.
"""
import mujoco
import mujoco.viewer
import numpy as np
import time

import example_utils as ex_utils
import mj_maniPlan.utils as utils
import panda_utils
import ruckig_trajectory as r_traj


if __name__ == '__main__':
    visualize, use_obstacles, seed = panda_utils.parse_panda_args("Generate and follow a trajectory from a RRT path with computed torque control.")
    model, _, shortcut_path, joint_qpos_addrs = panda_utils.rrt_panda(use_obstacles, seed)

    print("Generating a trajectory on the shortcut path...")
    start = time.time()
    traj = r_traj.generate_trajectory(len(shortcut_path[0]), model.opt.timestep, shortcut_path)
    print(f"Trajectory generation took {(time.time() - start):.4f}s")

    if visualize:
        # Since we are doing computed torque control, we need to "disable" position actuators
        # so that data.ctrl is not part of the applied force when we step simulation forward.
        model.actuator_gainprm[:] = 0
        model.actuator_biasprm[:] = 0
        data = mujoco.MjData(model)
        q_init, q_goal = shortcut_path[0], shortcut_path[-1]
        with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
            # Update the viewer's orientation to capture the scene.
            viewer.cam.lookat = [0, 0, 0.35]
            viewer.cam.distance = 2.5
            viewer.cam.azimuth = 145
            viewer.cam.elevation = -25

            # Draw the initial and target EE pose.
            ex_utils.add_site_frame(viewer.user_scn, model, panda_utils._PANDA_EE_SITE, q_init, joint_qpos_addrs)
            ex_utils.add_site_frame(viewer.user_scn, model, panda_utils._PANDA_EE_SITE, q_goal, joint_qpos_addrs)

            # Show the shortcut path.
            ex_utils.add_path(viewer.user_scn, model, panda_utils._PANDA_EE_SITE, joint_qpos_addrs, shortcut_path, [0.2, 0.6, 0.2, 0.2])

            # Ensure the robot is at q_init and then update the viewer to show
            # the frames, paths, and initial state.
            utils.fk(q_init, joint_qpos_addrs, model, data)
            viewer.sync()

            # Command the robot along the path via computed torque control.
            while viewer.is_running():
                time.sleep(0.5)

                # Reset to initial state of trajectory
                mujoco.mj_resetData(model, data)
                data.qpos[joint_qpos_addrs] = traj.configurations[0]
                data.qvel[joint_qpos_addrs] = traj.velocities[0]
                mujoco.mj_forward(model, data)

                for t in range(len(traj.configurations)):
                    start_time = time.time()

                    # computed torque control with PD feedback for position/velocity error
                    error = traj.configurations[t] - data.qpos[joint_qpos_addrs]
                    error_dot = traj.velocities[t] - data.qvel[joint_qpos_addrs]
                    # NOTE: these gains may need to be tuned
                    Kp = np.eye(len(joint_qpos_addrs)) * 1.0
                    Kd = np.eye(len(joint_qpos_addrs)) * 1.0
                    data.qacc[:] = 0
                    data.qacc[joint_qpos_addrs] = traj.accelerations[t] + Kd.dot(error_dot) + Kp.dot(error)
                    mujoco.mj_inverse(model, data)
                    data.qfrc_applied[joint_qpos_addrs] = data.qfrc_inverse[joint_qpos_addrs]
                    mujoco.mj_step(model, data)

                    duration = time.time() - start_time
                    if duration < traj.dt:
                        time.sleep(traj.dt - duration)
                    viewer.sync()
