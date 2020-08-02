import numpy as np


def mean_square(points: np.ndarray):
	mean = np.mean(points, axis=(0, 1))
	var = np.mean(abs(points - mean)**2)
	return var
