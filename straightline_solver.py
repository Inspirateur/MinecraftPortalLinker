import numpy as np
from scipy.spatial import Voronoi


def intersect(p: np.ndarray, c: np.ndarray, rlines: np.ndarray) -> np.ndarray:
	for line in rlines:
		...
	return p


def reg_lines(vor) -> np.ndarray:
	# compute the lines delimiting each region
	rlines = [[] for _ in range(len(vor.points))]
	center = vor.points.mean(axis=0)
	ptp_bound = vor.points.ptp(axis=0)
	for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
		reg_i = []
		for r, i in enumerate(vor.point_region):
			if all(j in vor.regions[i] for j in simplex):
				reg_i.append(r)
		simplex = np.asarray(simplex)
		if np.all(simplex >= 0):
			for r in reg_i:
				rlines[r].append(vor.vertices[simplex])
		else:
			i = simplex[simplex >= 0][0]  # finite end Voronoi vertex
			t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
			t /= np.linalg.norm(t)
			n = np.array([-t[1], t[0]])  # normal
			midpoint = vor.points[pointidx].mean(axis=0)
			direction = np.sign(np.dot(midpoint - center, n)) * n
			if vor.furthest_site:
				direction = -direction
			far_point = vor.vertices[i] + direction * ptp_bound.max()
			for r in reg_i:
				rlines[r].append([vor.vertices[i], far_point])

	return rlines


def solve(vor: Voronoi):
	rlines = reg_lines(vor)
	# iterate the solution
	res = np.array(vor.points)
	# compute the center C of current solution
	center = res.mean(axis=0)
	# for each point P in the solution
	for i in range(len(res)):
		# compute the intersection between PC and the edges of P's region
		res[i] = intersect(res[i], center, rlines[i])
	return res
