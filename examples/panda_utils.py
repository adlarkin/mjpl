"""
Utilities used for example scripts that use the Franka Panda MuJoCo models.
"""
import argparse
import mujoco
import numpy as np
import time
from pathlib import Path

import mj_maniPlan.utils as utils
from mj_maniPlan.rrt import (
    RRT,
    RRTOptions,
)


_HERE = Path(__file__).parent

_PANDA_XML = _HERE.parent / "models" / "franka_emika_panda" / "scene.xml"
_PANDA_OBSTACLES_XML = _HERE.parent / "models" / "franka_emika_panda" / "scene_with_obstacles.xml"
_PANDA_EE_SITE = 'ee_site'


def load_panda_model(include_obstacles: bool) -> mujoco.MjModel:
    if include_obstacles:
        return mujoco.MjModel.from_xml_path(_PANDA_OBSTACLES_XML.as_posix())
    return mujoco.MjModel.from_xml_path(_PANDA_XML.as_posix())

def panda_arm_joints() -> list[str]:
    return [
        'joint1',
        'joint2',
        'joint3',
        'joint4',
        'joint5',
        'joint6',
        'joint7',
    ]

def parse_panda_args(description: str):
    parser = argparse.ArgumentParser(
        description=description
    )
    parser.add_argument(
        "-viz",
        "--visualize",
        action="store_true",    # set to True if flag is provided
        default=False,          # default value if flag is not provided
        help="Visualize paths via the mujoco viewer"
    )
    parser.add_argument(
        "-obs",
        "--obstacles",
        action="store_true",    # set to True if flag is provided
        default=False,          # default value if flag is not provided
        help="Use obstacles in the environment"
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=-1,
        help="Seed for random sampling. Must be >= 0. If not set, a random seed will be used"
    )
    args = parser.parse_args()
    seed = args.seed
    if seed < 0:
        seed = None
    return args.visualize, args.obstacles, seed

def rrt_panda(use_obstacles: bool, seed: int | None):
    model = load_panda_model(include_obstacles=use_obstacles)
    data = mujoco.MjData(model)

    # The joints to sample during planning.
    # Since this example executes planning for the arm,
    # the finger joints of the gripper are excluded.
    joint_names = panda_arm_joints()

    # Use the "home" configuration as q_init.
    joint_qpos_addrs = utils.joint_names_to_qpos_addrs(joint_names, model)
    q_init = model.key('home').qpos[joint_qpos_addrs]
    # Generate valid goal configuration.
    lower_limits, upper_limits = utils.joint_limits(joint_names, model)
    rng = np.random.default_rng(seed=seed)
    q_goal = utils.random_valid_config(rng, lower_limits, upper_limits, joint_qpos_addrs, model, data)

    # Set up the planner.
    epsilon = 0.05
    planner_options = RRTOptions(
        joint_names=joint_names,
        max_planning_time=10,
        epsilon=epsilon,
        shortcut_filler_epsilon=10*epsilon,
        seed=seed,
        goal_biasing_probability=0.1,
    )
    planner = RRT(planner_options, model)

    print("Planning...")
    start = time.time()
    path = planner.plan(q_init, q_goal)
    if not path:
        print("Planning failed")
        return
    print(f"Planning took {(time.time() - start):.4f}s")

    print("Shortcutting...")
    start = time.time()
    shortcut_path = planner.shortcut(path, num_attempts=len(path))
    print(f"Shortcutting took {(time.time() - start):.4f}s")

    return model, path, shortcut_path, joint_qpos_addrs
