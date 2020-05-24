import yaml
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import matplotlib.pyplot as plt
from straightline_solver import solve


def plot_voronoi():
	with open("portals.yaml", "r") as fp:
		portals = yaml.load(fp)
	points = np.array([p for p in portals.values()])/8
	vor = Voronoi(points)
	res = solve(vor)
	print(f"loss {loss(res):.1f}")
	voronoi_plot_2d(vor)
	plt.scatter(res[:, 0], res[:, 1], color='red')
	for i, name in enumerate(portals.keys()):
		plt.annotate(name, (res[i, 0], res[i, 1]+0.06*(vor.min_bound[0]-vor.max_bound[0])), ha="center")
	plt.show()


def loss(points: np.ndarray):
	return points.var()/len(points)


if __name__ == '__main__':
	plot_voronoi()
