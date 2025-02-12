"""
Utilities that are used in example scripts.
"""

import mujoco
import numpy as np
from scipy.interpolate import make_interp_spline

import mj_maniPlan.utils as utils
import mj_maniPlan.visualization as viz
from mj_maniPlan.configuration import Configuration


def fit_path_to_spline(
    path: list[np.ndarray], interval: tuple[float, float] = (0.0, 1.0)
):
    # Create "timing" for the path, which is used for B-spline interpolation.
    # Configuration distance between two adjacent path waypoints - q_curr, q_next - is used as a notion
    # for the time it takes to move from q_curr to q_next.
    timing = [0.0]
    for i in range(1, len(path)):
        timing.append(timing[-1] + utils.configuration_distance(path[i - 1], path[i]))
    # scale to the interval bounds
    timing = np.interp(timing, (timing[0], timing[-1]), interval)

    return make_interp_spline(timing, path)


def add_path(scene: mujoco.MjvScene, site: str, path, config: Configuration, rgba):
    data = mujoco.MjData(config.model)
    spl_x_bounds = (0, 1)
    spline = fit_path_to_spline(path, interval=spl_x_bounds)
    horizon = np.linspace(spl_x_bounds[0], spl_x_bounds[1], 1000)
    for t in horizon:
        q_t = spline(t)
        config.fk(q_t, data)
        # Use a sphere at the site's world position to show the current state of the path.
        world_pos = data.site(site).xpos
        viz.add_sphere(scene, world_pos, 0.004, rgba)


def add_site_frame(scene: mujoco.MjvScene, site: str, q, config: Configuration):
    data = mujoco.MjData(config.model)
    config.fk(q, data)
    (
        pos,
        rot,
    ) = utils.site_pose(site, data)
    viz.add_frame(scene, pos, rot)
