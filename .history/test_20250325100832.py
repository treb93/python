import numpy as np

def normalize_position(
    position: np.ndarray, min: np.ndarray, max: np.ndarray
) -> np.ndarray:
    return (position - min) / (max - min)


def denormalize_position(
    position: np.ndarray, min: np.ndarray, max: np.ndarray
) -> np.ndarray:
    return min + position * (max - min)


def get_x(
    center: np.ndarray,
    area: float,
    ratio: float,
    o: float,
) -> np.ndarray:
    height_width_ratio = np.exp(ratio * 2)
    width = (area / height_width_ratio) ** 0.5
    height = area / width

    x = np.array(
        [
            [-width, -height],
            [-width, height],
            [width, height],
            [width, -height],
        ]
    )

    x *= 0.5

    x = rotate_points(x, o)

    return x + center


def rotate_points(points: np.ndarray, o: float) -> np.ndarray:
    angle = o * 2 * np.pi
    cos = np.cos(angle)
    sin = np.sin(angle)
    rotation = np.array([[cos, -sin], [sin, cos]])

    vectorized_dot = np.vectorize(lambda x: np.dot(rotation, x))
    return vectorized_dot(points)
