import numpy as np
import jax

def vector_to_grid_position(vector: np.ndarray, resolution: int, rounded=True):
    if rounded:
        grid_position = np.floor(vector * resolution).astype(int)
        grid_position = np.clip(grid_position, 0, resolution - 1)
    else:
        grid_position = np.clip(
            (vector * resolution) - 0.5, 0, resolution - 1
        )

    return grid_position


def grid_position_to_vector(grid_position: np.ndarray, resolution: int) -> np.ndarray:
    return np.array(grid_position + 0.5) / resolution


def vector_to_map(vector: np.ndarray, resolution: int) -> np.ndarray:
    grid_position = vector_to_grid_position(vector, resolution)

    map = np.zeros((resolution, resolution)).astype(float)

    map[grid_position[0], grid_position[1]] = 1

    return map


def map_to_vector(map: np.ndarray):
    assert map.shape[0] == map.shape[1]
    
    resolution = map.shape[0]
    
    flatten_map = map.reshape(-1)
    flatten_position = np.argmax(flatten_map).astype(int)
    
    grid_position = np.array(
        [
            flatten_position // resolution,
            flatten_position % resolution,
        ]
    )

    return grid_position_to_vector(grid_position, resolution)


def function_1(vector: np.ndarray, resolution: int) -> np.ndarray:
    grid_position = vector_to_grid_position(vector, resolution, False)
    
    logit = grid_position[0] * resolution + grid_position[1]
    
    flatten_map = jax.nn.one_hot(logit, resolution ** 2)
    
    return flatten_map.reshape((resolution, resolution))
    

def normalize_position(
    position: np.ndarray, min: np.ndarray, max: np.ndarray
) -> np.ndarray:
    return (position - min) / (max - min)


def denormalize_position(
    position: np.ndarray, min: np.ndarray, max: np.ndarray
) -> np.ndarray:
    return min + position * (max - min)


def rotate_points(points: np.ndarray, o: float) -> np.ndarray:
    angle = o * 2 * np.pi
    cos = np.cos(angle)
    sin = np.sin(angle)
    rotation = np.array([[cos, -sin], [sin, cos]])

    vectorized_dot = np.vectorize(lambda x: np.dot(rotation, x))
    return vectorized_dot(points)
