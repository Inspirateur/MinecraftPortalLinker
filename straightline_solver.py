import numpy as np
from scipy.spatial import Voronoi


def solve(vpoints: Voronoi):
	res = np.array(vpoints.points)
	for i in range(1):
		center = res.mean(axis=0)
		pass
	return res
