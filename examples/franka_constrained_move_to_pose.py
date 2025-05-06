import time
from pathlib import Path

import example_utils as ex_utils
import mujoco
import mujoco.viewer
import numpy as np

import mjpl
import mjpl.visualization as viz

_HERE = Path(__file__).parent
_PANDA_XML = _HERE / "models" / "franka_emika_panda" / "scene_with_obstacles.xml"
_PANDA_EE_SITE = "ee_site"


def main():
    visualize, seed = ex_utils.parse_args(
        description="Compute and follow a trajectory to a goal pose."
    )

    model = mujoco.MjModel.from_xml_path(_PANDA_XML.as_posix())

    arm_joints = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    q_idx = mjpl.qpos_idx(model, arm_joints)

    epsilon = 0.05

    # Let the "home" keyframe in the MJCF be the initial state.
    home_keyframe = model.keyframe("home")
    q_init = home_keyframe.qpos.copy()

    # Constrain the end effector's pose to only rotate a limited amount about the
    # x and y axis of the end effector's initial pose. This has the effect of keeping
    # the gripper pointed "down".
    data = mujoco.MjData(model)
    data.qpos = q_init
    mujoco.mj_kinematics(model, data)
    ee_init_pose = mjpl.site_pose(data, _PANDA_EE_SITE)
    ee_pose_constraint = mjpl.PoseConstraint(
        model,
        (-np.inf, np.inf),
        (-np.inf, np.inf),
        (-np.inf, np.inf),
        (-0.1, 0.1),
        (-0.1, 0.1),
        (-np.inf, np.inf),
        ee_init_pose,
        _PANDA_EE_SITE,
        tolerance=0.001,
        q_step=epsilon,
    )

    allowed_collisions = [("left_finger", "right_finger")]
    constraints = [
        ee_pose_constraint,
        mjpl.JointLimitConstraint(model),
        mjpl.CollisionConstraint(model, allowed_collisions),
    ]

    # From the initial state, generate a valid goal pose that's derived from a
    # valid joint configuration.
    mujoco.mj_resetDataKeyframe(model, data, home_keyframe.id)
    data.qpos = mjpl.random_config(model, q_init, arm_joints, seed, constraints)
    mujoco.mj_kinematics(model, data)
    goal_pose = mjpl.site_pose(data, _PANDA_EE_SITE)

    # Set up the planner.
    planner = mjpl.RRT(
        model, arm_joints, constraints, seed=seed, goal_biasing_probability=0.1
    )

    print("Planning...")
    start = time.time()
    waypoints = planner.plan_to_pose(q_init, goal_pose, _PANDA_EE_SITE)
    if not waypoints:
        print("Planning failed")
        return
    print(f"Planning took {(time.time() - start):.4f}s")

    print("Shortcutting...")
    start = time.time()
    shortcut_waypoints = mjpl.shortcut(
        waypoints,
        constraints,
        validation_dist=planner.epsilon,
        max_attempts=len(waypoints),
        seed=seed,
    )
    print(f"Shortcutting took {(time.time() - start):.4f}s")

    # The trajectory limits used here are for demonstration purposes only.
    # In practice, consult your hardware spec sheet for this information.
    dof = len(waypoints[0])
    traj_generator = mjpl.RuckigTrajectoryGenerator(
        dt=model.opt.timestep,
        max_velocity=np.ones(dof) * np.pi,
        max_acceleration=np.ones(dof) * 0.5 * np.pi,
        max_jerk=np.ones(dof),
    )

    print("Generating trajectory...")
    start = time.time()
    trajectory = mjpl.generate_constrained_trajectory(
        shortcut_waypoints, traj_generator, constraints
    )
    print(f"Trajectory generation took {(time.time() - start):.4f}s")

    # Actuator indices in data.ctrl that correspond to the joints in the trajectory.
    actuators = [
        "actuator1",
        "actuator2",
        "actuator3",
        "actuator4",
        "actuator5",
        "actuator6",
        "actuator7",
    ]
    actuator_ids = [model.actuator(act).id for act in actuators]

    # Follow the trajectory via position control, starting from the initial state.
    mujoco.mj_resetDataKeyframe(model, data, home_keyframe.id)
    q_t = [q_init]
    for q_ref in trajectory.positions:
        data.ctrl[actuator_ids] = q_ref[q_idx]
        mujoco.mj_step(model, data)
        q_t.append(data.qpos.copy())

    if visualize:
        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            # Update the viewer's orientation to capture the scene.
            viewer.cam.lookat = [0, 0, 0.35]
            viewer.cam.distance = 2.5
            viewer.cam.azimuth = 145
            viewer.cam.elevation = -25

            # Visualize the initial EE pose.
            data.qpos = q_init
            mujoco.mj_kinematics(model, data)
            initial_pose = mjpl.site_pose(data, _PANDA_EE_SITE)
            viz.add_frame(
                viewer.user_scn,
                initial_pose.translation(),
                initial_pose.rotation().as_matrix(),
            )

            # Visualize the goal EE pose.
            viz.add_frame(
                viewer.user_scn,
                goal_pose.translation(),
                goal_pose.rotation().as_matrix(),
            )

            # Visualize the trajectory. The trajectory is of high resolution,
            # so plotting every other timestep should be sufficient.
            for q_ref in trajectory.positions[::2]:
                data.qpos = q_ref
                mujoco.mj_kinematics(model, data)
                pos = data.site(_PANDA_EE_SITE).xpos
                viz.add_sphere(
                    viewer.user_scn, pos, radius=0.004, rgba=[0.2, 0.6, 0.2, 0.2]
                )

            # Replay the robot following the trajectory.
            for q_actual in q_t:
                start_time = time.time()
                if not viewer.is_running():
                    return
                data.qpos = q_actual
                mujoco.mj_kinematics(model, data)
                viewer.sync()
                time_until_next_step = model.opt.timestep - (time.time() - start_time)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
