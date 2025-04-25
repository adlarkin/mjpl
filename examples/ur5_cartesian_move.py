import time
from pathlib import Path

import example_utils as ex_utils
import mujoco
import mujoco.viewer
import numpy as np
from mink.lie import SE3

import mjpl
import mjpl.visualization as viz

_HERE = Path(__file__).parent
_UR5_XML = _HERE / "models" / "universal_robots_ur5e" / "scene.xml"
_UR5_EE_SITE = "attachment_site"


def circle_waypoints(
    radius: float, c_x: float, c_y: float, num_points: int = 25
) -> np.ndarray:
    """Create waypoints that form a circle centered about (c_x,c_y)"""
    t = np.linspace(0, 1, num_points)
    x = radius * np.cos(2 * np.pi * t) + c_x
    y = radius * np.sin(2 * np.pi * t) + c_y
    return np.stack((x, y), axis=1)


def main():
    visualize, seed = ex_utils.parse_args(
        description="Compute and follow a trajectory along a cartesian path."
    )

    model = mujoco.MjModel.from_xml_path(_UR5_XML.as_posix())
    data = mujoco.MjData(model)

    arm_joints = mjpl.all_joints(model)
    q_idx = mjpl.qpos_idx(model, arm_joints)

    cr = mjpl.CollisionRuleset()

    # Let the "home" keyframe in the MJCF be the initial state.
    home_keyframe = model.keyframe("home")
    q_init = home_keyframe.qpos.copy()

    # Define a cartesian path that corresponds to the EE moving in a circle
    # in the xy plane, centered about the initial EE pose
    mujoco.mj_resetDataKeyframe(model, data, home_keyframe.id)
    mujoco.mj_kinematics(model, data)
    initial_ee_pose = mjpl.site_pose(data, _UR5_EE_SITE)
    ee_x, ee_y, ee_z = initial_ee_pose.translation()
    poses = [
        SE3.from_rotation_and_translation(
            initial_ee_pose.rotation(), np.array([x, y, ee_z])
        )
        for x, y in circle_waypoints(radius=0.1, c_x=ee_x, c_y=ee_y)
    ]

    solver = mjpl.MinkIKSolver(
        model=model,
        joints=arm_joints,
        cr=cr,
        seed=seed,
        max_attempts=5,
    )

    print("Planning...")
    start = time.time()
    path = mjpl.cartesian_plan(model, q_init, poses, _UR5_EE_SITE, solver)
    if path is None:
        print("Planning failed")
        return
    print(f"Planning took {(time.time() - start):.4f}s")

    # The trajectory limits used here are for demonstration purposes only.
    # In practice, consult your hardware spec sheet for this information.
    dof = len(path.waypoints[0])
    traj_generator = mjpl.ToppraTrajectoryGenerator(
        dt=model.opt.timestep,
        max_velocity=np.ones(dof) * 0.5 * np.pi,
        max_acceleration=np.ones(dof) * 0.25 * np.pi,
    )

    print("Generating trajectory...")
    start = time.time()
    trajectory = traj_generator.generate_trajectory(path)
    print(f"Trajectory generation took {(time.time() - start):.4f}s")

    # Actuator indices in data.ctrl that correspond to the joints in the trajectory.
    actuators = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ]
    actuator_ids = [model.actuator(act).id for act in actuators]

    # Follow the trajectory via position control, starting from the initial state.
    mujoco.mj_resetDataKeyframe(model, data, home_keyframe.id)
    q_t = [q_init]
    for q_ref in trajectory.positions:
        data.ctrl[actuator_ids] = q_ref
        mujoco.mj_step(model, data)
        q_t.append(data.qpos.copy())

    if visualize:
        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            # Update the viewer's orientation to capture the scene.
            viewer.cam.lookat = [-0.1, 0, 0.35]
            viewer.cam.distance = 1.5
            viewer.cam.azimuth = -90
            viewer.cam.elevation = -10

            # Add a marker for each pose in the cartesian path.
            data.qpos = q_init
            mujoco.mj_kinematics(model, data)
            for p in poses:
                viz.add_sphere(
                    viewer.user_scn,
                    p.translation(),
                    radius=0.004,
                    rgba=[0.6, 0.2, 0.2, 0.7],
                )

            # Visualize the trajectory. The trajectory is of high resolution,
            # so plotting every other timestep should be sufficient.
            for q_ref in trajectory.positions[::2]:
                data.qpos[q_idx] = q_ref
                mujoco.mj_kinematics(model, data)
                pos = data.site(_UR5_EE_SITE).xpos
                viz.add_sphere(
                    viewer.user_scn, pos, radius=0.002, rgba=[0.2, 0.6, 0.2, 0.2]
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
