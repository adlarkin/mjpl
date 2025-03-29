import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from example_utils import parse_args
from mink.lie import SE3

import mj_maniPlan.utils as utils
import mj_maniPlan.visualization as viz
from mj_maniPlan.cartesian_planner import cartesian_plan
from mj_maniPlan.collision_ruleset import CollisionRuleset
from mj_maniPlan.inverse_kinematics.mink_ik_solver import MinkIKSolver
from mj_maniPlan.joint_group import JointGroup
from mj_maniPlan.trajectory import TrajectoryLimits, generate_trajectory

_HERE = Path(__file__).parent
_UR5_XML = _HERE / "models" / "universal_robots_ur5e" / "scene.xml"
_UR5_EE_SITE = "attachment_site"


def circle_waypoints(
    radius: float, h: float, k: float, num_points: int = 10
) -> np.ndarray:
    """Create waypoints that form a circle centered about (h,k)"""
    t = np.linspace(0, 1, num_points)
    x = radius * np.cos(2 * np.pi * t) + h
    y = radius * np.sin(2 * np.pi * t) + k
    return np.stack((x, y), axis=1)


def main():
    visualize, seed = parse_args(
        description="Compute and follow a trajectory along a cartesian path."
    )

    model = mujoco.MjModel.from_xml_path(_UR5_XML.as_posix())
    data = mujoco.MjData(model)

    arm_joints = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    arm_joint_ids = [model.joint(joint).id for joint in arm_joints]
    arm_jg = JointGroup(model, arm_joint_ids)

    cr = CollisionRuleset(model)

    # Let the "home" keyframe in the MJCF be the initial state.
    home_keyframe = model.keyframe("home")
    q_init = home_keyframe.qpos.copy()

    # Define a cartesian path that corresponds to the EE moving in a circle
    # in the xy plane, centered about the initial EE pose
    mujoco.mj_resetDataKeyframe(model, data, home_keyframe.id)
    mujoco.mj_kinematics(model, data)
    initial_ee_pose = utils.site_pose(data, _UR5_EE_SITE)
    x, y, z = initial_ee_pose.translation()
    poses = [
        SE3.from_rotation_and_translation(
            initial_ee_pose.rotation(), np.array([c_x, c_y, z])
        )
        for c_x, c_y in circle_waypoints(radius=0.1, h=x, k=y)
    ]

    solver = MinkIKSolver(
        model=model,
        jg=arm_jg,
        cr=cr,
        seed=seed,
        max_attempts=5,
    )

    print("Planning...")
    start = time.time()
    path = cartesian_plan(q_init, poses, _UR5_EE_SITE, solver)
    if not path:
        print("Planning failed")
        return
    print(f"Planning took {(time.time() - start):.4f}s")

    # Cartesian plan gives full world configuration.
    # We only need the joints in the JointGroup for trajectory generation.
    path_jg = [q[arm_jg.qpos_addrs] for q in path]

    # These values are for demonstration purposes only.
    # In practice, consult your hardware spec sheet for this information.
    dof = len(arm_joints)
    tr_limits = TrajectoryLimits(
        jg=arm_jg,
        min_velocity=-np.ones(dof) * np.pi,
        max_velocity=np.ones(dof) * np.pi,
        min_acceleration=-np.ones(dof) * 0.5 * np.pi,
        max_acceleration=np.ones(dof) * 0.5 * np.pi,
        jerk=np.ones(dof),
    )

    print("Generating trajectory...")
    start = time.time()
    traj = generate_trajectory(path_jg, tr_limits, model.opt.timestep)
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
    for q_ref in traj.configurations:
        data.ctrl[actuator_ids] = q_ref
        mujoco.mj_step(model, data)
        # While the planner gives a sequence of waypoints are collision free, the
        # generated trajectory may not. For more info, see:
        # https://github.com/adlarkin/mj_maniPlan/issues/54
        if not cr.obeys_ruleset(data.contact.geom):
            print("Invalid collision occurred during trajectory execution.")
            return
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
            for q_ref in traj.configurations[::2]:
                arm_jg.fk(q_ref, data)
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
