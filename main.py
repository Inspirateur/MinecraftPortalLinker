from time import time
import yaml
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_solve(solve):
	with open("portals.yaml", "r") as fp:
		portals = yaml.load(fp)
	points = np.array([p for p in portals.values()])/8
	vor = Voronoi(points)
	start = time()
	res = solve(vor)
	delta = time()-start
	maxlen = max(len(name) for name in portals.keys())+1
	for name, p in zip(portals.keys(), res):
		print(f"{(name+':').ljust(maxlen)} {p[0]:.1f};{p[1]:.1f}")
	print(f"loss {loss(res):.1f} (in {int(delta*1000)} ms)")
	fig, ax = plt.subplots()
	voronoi_plot_2d(vor, ax, show_vertices=False)
	plt.scatter(res[:, 0], res[:, 1], marker="x", color='red', zorder=10)
	for i, name in enumerate(portals.keys()):
		plt.annotate(name, (points[i, 0], points[i, 1]+0.03*(vor.min_bound[0]-vor.max_bound[0])), ha="center")
	plt.title("Optimal portal repartition")
	plt.ylabel("z")
	plt.xlabel("x")
	plt.gca().invert_yaxis()
	plt.gca().set_aspect('equal', adjustable='box')
	plt.grid(linestyle="--")
	ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*8))
	ax.xaxis.set_major_formatter(ticks)
	ax.xaxis.set_major_locator(plt.MultipleLocator(50.0/8))
	ax.yaxis.set_major_formatter(ticks)
	ax.yaxis.set_major_locator(plt.MultipleLocator(50.0/8))
	plt.show()


def loss(points: np.ndarray):
	return points.var()/len(points)
