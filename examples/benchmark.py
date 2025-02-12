"""
Benchmark for testing planning time.

The testing methodology is as follows:
1. To be deterministic, seed the random number generator.
2. Create a planner and a [q_init, q_goal] pairing.
3. Plan from q_init to q_goal N number of times.
4. Report the following:
    a. How many plans succeeded vs how many plans timed out (success rate)
    b. Median planning time of successful planning attempts
"""

import time

import mujoco
import numpy as np
import panda_utils

import mj_maniPlan.utils as utils
from mj_maniPlan.configuration import Configuration
from mj_maniPlan.rrt import (
    RRT,
    RRTOptions,
)

if __name__ == "__main__":
    # NOTE: modify these parameters as needed for your benchmarking needs.
    joint_names = panda_utils.panda_arm_joints()
    max_planning_time = 10
    epsilon = 0.05
    seed = 42  # 5
    goal_biasing_probability = 0.1
    number_of_attempts = 15

    model = panda_utils.load_panda_model(include_obstacles=False)

    config = Configuration(joint_names, model)

    # Plan number_of_attempts times and record benchmarks.
    successful_planning_times = []
    for i in range(number_of_attempts):
        # Generate a valid initial and goal configuration.
        data = mujoco.MjData(model)
        rng = np.random.default_rng(seed=seed)
        q_init = utils.random_valid_config(rng, config, data)
        q_goal = utils.random_valid_config(rng, config, data)

        planner_options = RRTOptions(
            joint_names=joint_names,
            max_planning_time=max_planning_time,
            epsilon=epsilon,
            shortcut_filler_epsilon=epsilon,
            seed=seed,
            goal_biasing_probability=goal_biasing_probability,
        )
        planner = RRT(planner_options, model)

        print(f"Attempt {i}...")
        start_time = time.time()
        path = planner.plan(q_init, q_goal)
        elapsed_time = time.time() - start_time
        if path:
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
