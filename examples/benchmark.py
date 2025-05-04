"""
Benchmark for testing planning time.

The testing methodology is as follows:
1. To be deterministic, seed the random number generator.
2. Create a planner and a [q_init, ee_goal_pose] pairing.
3. Plan from q_init to ee_goal_pose N number of times.
4. Report the following:
    a. How many plans succeeded vs how many plans timed out (success rate)
    b. Median planning time of successful planning attempts
"""

import time
from pathlib import Path

import mujoco
import numpy as np

import mjpl

_HERE = Path(__file__).parent
_PANDA_XML = _HERE / "models" / "franka_emika_panda" / "scene.xml"
_PANDA_EE_SITE = "ee_site"


if __name__ == "__main__":
    # NOTE: modify these parameters as needed for your benchmarking needs.
    model = mujoco.MjModel.from_xml_path(_PANDA_XML.as_posix())
    planning_joints = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    allowed_collisions = [("left_finger", "right_finger")]
    max_planning_time = 10
    epsilon = 0.05
    seed = 42
    goal_biasing_probability = 0.1
    number_of_attempts = 15

    constraints = [
        mjpl.JointLimitConstraint(model),
        mjpl.CollisionConstraint(model),
    ]

    # Plan number_of_attempts times and record benchmarks.
    successful_planning_times = []
    for i in range(number_of_attempts):
        # Let the "home" keyframe in the MJCF be the initial state.
        home_keyframe = model.keyframe("home")
        q_init = home_keyframe.qpos.copy()

        # From the initial state, generate a goal pose.
        data = mujoco.MjData(model)
        data.qpos = mjpl.random_config(
            model, q_init, planning_joints, seed, constraints
        )
        mujoco.mj_kinematics(model, data)
        goal_pose = mjpl.site_pose(data, _PANDA_EE_SITE)

        planner = mjpl.RRT(
            model,
            planning_joints,
            constraints,
            max_planning_time=max_planning_time,
            epsilon=epsilon,
            seed=seed,
            goal_biasing_probability=goal_biasing_probability,
        )

        print(f"Attempt {i}...")
        start_time = time.time()
        waypoints = planner.plan_to_pose(q_init, goal_pose, _PANDA_EE_SITE)
        elapsed_time = time.time() - start_time
        if waypoints:
            successful_planning_times.append(elapsed_time)
    print()

    successful_attempts = len(successful_planning_times)
    success_rate = successful_attempts / number_of_attempts
    median_planning_time = np.median(successful_planning_times)
    print(
        f"Attempted {number_of_attempts} plans, succeeded on {successful_attempts} attempts (success rate of {success_rate:.2f})"
    )
    print(
        f"Median planning time of successful plans: {median_planning_time:.4f} seconds"
    )
