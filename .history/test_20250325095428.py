import numpy as np


def normalize_position(
    position: np.ndarray, min: np.ndarray, max: np.ndarray
) -> np.ndarray:
    return (position - min) / (max - min)


def denormalize_position(
    position: np.ndarray, min: np.ndarray, max: np.ndarray
) -> np.ndarray:
    return min + position * (max - min)
