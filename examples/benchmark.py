'''
Benchmark for testing planning time.

The testing methodology is as follows:
1. To be deterministic, seed the random number generator.
2. Create a planner and a [q_init, q_goal] pairing.
3. Plan from q_init to q_goal N number of times.
4. Report the following:
    a. How many plans succeeded vs how many plans timed out (success rate)
    b. Median planning time of successful planning attempts
'''

import mujoco
import numpy as np
import time

from mj_maniPlan.rrt import (
    RRT,
    RRTOptions,
)
import mj_maniPlan.utils as utils


if __name__ == '__main__':
    # NOTE: modify these parameters as needed for your benchmarking needs.
    model_xml_path = 'models/franka_emika_panda/scene.xml'
    # The joints to sample during planning.
    joint_names = [
        'joint1',
        'joint2',
        'joint3',
        'joint4',
        'joint5',
        'joint6',
        'joint7',
    ]
    max_planning_time = 10
    epsilon = 0.05
    #seed = 5
    seed = 42
    goal_biasing_probability = 0.1
    number_of_attempts = 15

    model = mujoco.MjModel.from_xml_path(model_xml_path)
    data = mujoco.MjData(model)

    # Random number generator that's used for sampling joint configurations.
    rng = np.random.default_rng(seed=seed)

    # Generate a valid initial and goal configuration.
    joint_qpos_addrs = utils.joint_names_to_qpos_addrs(joint_names, model)
    lower_limits, upper_limits = utils.joint_limits(joint_names, model)
    q_init = utils.random_valid_config(rng, lower_limits, upper_limits, joint_qpos_addrs, model, data)
    q_goal = utils.random_valid_config(rng, lower_limits, upper_limits, joint_qpos_addrs, model, data)

    # Define the planner options.
    planner_options = RRTOptions(
        joint_names=joint_names,
        max_planning_time=max_planning_time,
        epsilon=epsilon,
        rng=rng,
        goal_biasing_probability=goal_biasing_probability,
    )

    # Plan number_of_attempts times and record benchmarks.
    # Since the planner sets q_init to the state of mujoco.MjData that was used during planner construction,
    # we recreate the planner with a mujoco.MjData that reflects q_init for each attempt.
    successful_planning_times = []
    for i in range(number_of_attempts):
        data.qpos[joint_qpos_addrs] = q_init
        mujoco.mj_kinematics(model, data)
        planner = RRT(planner_options, model, data)

        print(f"Attempt {i}...")
        start_time = time.time()
        path = planner.plan(q_goal)
        elapsed_time = time.time() - start_time
        if path:
            successful_planning_times.append(elapsed_time)
    print()

    successful_attempts = len(successful_planning_times)
    success_rate = successful_attempts / number_of_attempts
    median_planning_time = np.median(successful_planning_times)
    print(f"Attempted {number_of_attempts} plans, succeeded on {successful_attempts} attempts (success rate of {success_rate})")
    print(f"Median planning time of successful plans: {median_planning_time} seconds")
