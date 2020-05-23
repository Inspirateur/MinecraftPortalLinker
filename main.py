import yaml
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import matplotlib.pyplot as plt
from straightline_solver import solve


def plot_voronoi():
	with open("portals.yaml", "r") as fp:
		portals = yaml.load(fp)
	points = np.array([p for p in portals.values()])
	vpoints = Voronoi(points)
	res = solve(vpoints)
	print(f"loss {loss(res):.1f}")
	voronoi_plot_2d(vpoints)
	plt.scatter(res[:, 0], res[:, 1], color='red')
	plt.show()


def loss(points: np.ndarray):
	return points.var()/len(points)


if __name__ == '__main__':
	plot_voronoi()
