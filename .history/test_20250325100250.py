import numpy as np
from jax import Array

def normalize_position(
    position: np.ndarray, min: np.ndarray, max: np.ndarray
) -> np.ndarray:
    return (position - min) / (max - min)


def denormalize_position(
    position: np.ndarray, min: np.ndarray, max: np.ndarray
) -> np.ndarray:
    return min + position * (max - min)


def rotate_points(points: Array, orientation_normalized: float) -> Array:
    angle = orientation_normalized * 2 * np.pi
    cos = np.cos(angle)
    sin = np.sin(angle)
    rotation = np.array([[cos, -sin], [sin, cos]])

    vectorized_dot = np.vectorize(lambda x: np.dot(rotation, x))
    return vectorized_dot(points)


def get_rectangle_from_scalars(
    center: Array,
    area: float,
    height_width_ratio_log: float,
    orientation_normalized: float,
) -> Array:
    height_width_ratio = np.exp(height_width_ratio_log * 2)
    width = (area / height_width_ratio) ** 0.5
    height = area / width

    rectangle = np.array(
        [
            [-width, -height],
            [-width, height],
            [width, height],
            [width, -height],
        ]
    )

    rectangle *= 0.5

    rectangle = rotate_points(rectangle, orientation_normalized)

    return rectangle + center
