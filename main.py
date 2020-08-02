from time import time
import yaml
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from loss import mean_square


def plot_solve(solve):
	with open("portals.yaml", "r") as fp:
		portals = yaml.load(fp)
	points = np.array([p for p in portals.values()])
	points = np.transpose(points, (2, 0, 1))
	points = np.transpose(np.array([points[0]/8, points[1], points[2]/8]), (1, 2, 0))
	xz_points = np.mean(np.array([points.T[2], points.T[0]]).T, axis=1)
	vor = Voronoi(xz_points)
	start = time()
	res = solve(points, list(portals.keys()))
	delta = time()-start
	maxlen = max(len(name) for name in portals.keys())+1
	for name, p in zip(portals.keys(), res):
		print(f"{(name+':').ljust(maxlen)} {p[0]} --> {p[1]}")
	print(f"loss {mean_square(res):.1f} (in {int(delta*1000)} ms)")
	fig, ax = plt.subplots()
	voronoi_plot_2d(vor, ax, show_vertices=False)
	res_2d = np.mean(np.array([res.T[2], res.T[0]]).T, axis=1)
	plt.scatter(res_2d[:, 0], res_2d[:, 1], marker="x", color='red', zorder=10)
	x_min, y_min = np.min(res_2d, axis=0)
	x_max, y_max = np.max(res_2d, axis=0)
	x_min2, x_max2 = plt.gca().get_xlim()
	y_min2, y_max2 = plt.gca().get_ylim()
	plt.xlim(min(x_min, x_min2), max(x_max, x_max2))
	plt.ylim(min(y_min, y_min2), max(y_max, y_max2))
	for i, name in enumerate(portals.keys()):
		plt.annotate(name, (xz_points[i, 0], xz_points[i, 1]+0.03*(vor.min_bound[0]-vor.max_bound[0])), ha="center")
	plt.title("Optimal portal repartition")
	plt.xlabel("z")
	plt.ylabel("x")
	plt.gca().set_aspect('equal', adjustable='box')
	plt.grid(linestyle="--")
	ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x))  # *8
	ax.xaxis.set_major_formatter(ticks)
	ax.xaxis.set_major_locator(plt.MultipleLocator(50.0/8))
	ax.yaxis.set_major_formatter(ticks)
	ax.yaxis.set_major_locator(plt.MultipleLocator(50.0/8))
	plt.show()
