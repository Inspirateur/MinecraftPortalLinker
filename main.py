from time import time
import yaml
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import matplotlib.pyplot as plt


def plot_solve(solve):
	with open("portals.yaml", "r") as fp:
		portals = yaml.load(fp)
	points = np.array([p for p in portals.values()])/8
	vor = Voronoi(points)
	start = time()
	res = solve(vor)
	delta = time()-start
	print(f"loss {loss(res):.1f} (in {int(delta*1000)} ms)")
	voronoi_plot_2d(vor)
	plt.scatter(res[:, 0], res[:, 1], color='red')
	for i, name in enumerate(portals.keys()):
		plt.annotate(name, (points[i, 0], points[i, 1]+0.06*(vor.min_bound[0]-vor.max_bound[0])), ha="center")
	plt.show()


def loss(points: np.ndarray):
	return points.var()/len(points)
