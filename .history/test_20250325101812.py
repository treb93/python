import numpy as np
import jax

def vector_to_grid(vector: np.ndarray, resolution: int, rounded=True):
    if rounded:
        grid_position = np.floor(vector * resolution).astype(int)
        grid_position = np.clip(grid_position, 0, resolution - 1)
    else:
        grid_position = np.clip(
            (vector * resolution) - 0.5, 0, resolution - 1
        )

    return grid_position


def grid_to_vector(position_grid: np.ndarray, resolution: int) -> np.ndarray:
    return np.array(position_grid + 0.5) / resolution


def flatten_grid(position_grid: np.ndarray) -> np.ndarray:
    return position_grid.reshape(-1)



def vector_to_map(vector: np.ndarray, resolution: int) -> np.ndarray:
    grid_position = vector_to_grid(vector, resolution)

    map = np.zeros((resolution, resolution)).astype(float)

    map[grid_position[0], grid_position[1]] = 1

    return map


def function_1(vector: np.ndarray, resolution: int) -> np.ndarray:
    grid_position = vector_to_grid(vector, resolution)
    
    logit = grid_position[0] * resolution + grid_position[1]
    
    one_hot = jax.nn.one_hot(logit, resolution ** 2)
    
    return one_hot.reshape((resolution, resolution))
    




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
