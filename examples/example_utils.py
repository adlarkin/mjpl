import mujoco
import numpy as np
from pathlib import Path
from scipy.interpolate import make_interp_spline

import mj_maniPlan.utils as utils


_HERE = Path(__file__).parent

_PANDA_XML = _HERE.parent / "models" / "franka_emika_panda" / "scene_with_obstacles.xml"
_PANDA_EE_SITE = 'ee_site'


def load_panda_model() -> mujoco.MjModel:
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

def fit_path_to_spline(path: list[np.ndarray], interval: tuple[float, float] = (0.0, 1.0)):
    # Create "timing" for the path, which is used for B-spline interpolation.
    # Configuration distance between two adjacent path waypoints - q_curr, q_next - is used as a notion
    # for the time it takes to move from q_curr to q_next.
    timing = [0.0]
    for i in range(1, len(path)):
        timing.append(timing[-1] + utils.configuration_distance(path[i-1], path[i]))
    # scale to the interval bounds
    timing = np.interp(timing, (timing[0], timing[-1]), interval)

    return make_interp_spline(timing, path)
