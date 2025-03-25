import numpy as np

def normalize_position(
    position: np.ndarray, min: np.ndarray, max: np.ndarray
) -> np.ndarray:
    return (position - min) / (max - min)


def denormalize_position(
    position: np.ndarray, min: np.ndarray, max: np.ndarray
) -> np.ndarray:
    return min + position * (max - min)



def get_rectangle_from_scalars(
    center: np.ndarray,
    area: float,
    ratio: float,
    o: float,
) -> np.ndarray:
    height_width_ratio = np.exp(ratio * 2)
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

    rectangle = rotate_points(rectangle, o)

    return rectangle + center


def rotate_points(points: np.ndarray, orientation_normalized: float) -> np.ndarray:
    angle = orientation_normalized * 2 * np.pi
    cos = np.cos(angle)
    sin = np.sin(angle)
    rotation = np.array([[cos, -sin], [sin, cos]])

    vectorized_dot = np.vectorize(lambda x: np.dot(rotation, x))
    return vectorized_dot(points)
