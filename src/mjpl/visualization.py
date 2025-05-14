import mujoco
import numpy as np

# Axis colors (RGBA): X (red), Y (green), Z (blue)
_FRAME_COLORS = {
    "x": np.array([1.0, 0.0, 0.0, 1.0]),
    "y": np.array([0.0, 1.0, 0.0, 1.0]),
    "z": np.array([0.0, 0.0, 1.0, 1.0]),
}
# Frame axis directions
_FRAME_AXES = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def add_frame(
    scene: mujoco.MjvScene,
    origin: np.ndarray,
    rot_mat: np.ndarray,
    axis_radius: float = 0.0075,
    axis_length: float = 0.0525,
) -> None:
    """Add a coordinate frame to the scene.

    Args:
        scene: The MuJoCo scene.
        origin: [x,y,z] position of the frame origin.
        rot_mat: 3x3 rotation matrix that specifies the frame orientation.
        axis_radius: The radius of the cylinder for each axis.
        axis_length: The length of the cylinder for each axis.
    """
    for axis, color in _FRAME_COLORS.items():
        # Rotate axis direction by the orientation
        direction = rot_mat @ _FRAME_AXES[axis]
        # Set the end point of the axis marker.
        end_point = origin + axis_length * direction

        assert scene.ngeom < scene.maxgeom
        scene.ngeom += 1
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom - 1],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.eye(3).flatten(),
            rgba=color,
        )
        mujoco.mjv_connector(
            scene.geoms[scene.ngeom - 1],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            width=axis_radius,
            from_=origin,
            to=end_point,
        )


def add_sphere(
    scene: mujoco.MjvScene, origin: np.ndarray, radius: float, rgba: np.ndarray
):
    """Add a sphere to the scene.

    Args:
        scene: The MuJoCo scene.
        origin: [x,y,z] position of the sphere origin.
        radius: Radius of the sphere.
        rgba: Color of the sphere, in RGBA format.
    """
    assert scene.ngeom < scene.maxgeom
    scene.ngeom += 1
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom - 1],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0.0, 0.0]),
        pos=origin,
        mat=np.eye(3).flatten(),
        rgba=rgba,
    )
