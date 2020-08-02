import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import matplotlib.ticker as ticker
import sys
from loss import mean_square
from calculations import build_network, build_region, project, possible_portails, checking


def solve(portals, names):
	m_0 = np.mean(portals, axis=(0, 1))
	print(f"initial average mass of portals = {m_0} \n --------------------")

	print(f"building network")
	network, network_shape = build_network(portals)
	print("network built \n --------------------")

	region = build_region(network, network_shape, portals)
	print("--------------------")

	possible_portals = possible_portails(network, network_shape, portals, region, names)
	print(" --------------------")

	m_new = m_0
	print("starting iterative projection of the mass center")
	for j in range(666):
		res_portals = project(possible_portals, m_new)
		m_new = np.mean(res_portals, axis=(0, 1))
		if (m_new == m_0).all():
			print("converged in ", j+1, " itérations")
			break
		m_0 = m_new
	else:
		print("did not converged in saucisse iterations")
	print(f"solution center = {m_new} \n  --------------------")
	# loss_map(region, network, n_points)
	checking(res_portals, portals, names)
	return res_portals


def loss_map(region, network, n_points):
	x, y = network.T
	len_network = len(network)
	skip = 1
	z = np.zeros((len_network-1)//skip + 1)
	res = [[] for _ in range(n_points)]
	for n in trange(0, len_network, skip, desc="loss map", file=sys.stdout):
		for i in range(n_points):
			res[i] = region[i][np.int(np.argmin(np.linalg.norm(np.array(region[i]) - network[n], axis=1)))]
		z[n//skip] = mean_square(np.array(res))

	ind_min = np.argmin(z)
	array_skip = np.arange(0, len_network - 1, skip)
	print(f" le minimum est atteind pour m = ({x[array_skip][ind_min]}, {y[array_skip][ind_min]})")

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_trisurf(x[array_skip], y[array_skip], z, linewidth=0.2, antialiased=True)
	plt.grid(linestyle="--")
	ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*8))
	ax.xaxis.set_major_formatter(ticks)
	ax.xaxis.set_major_locator(plt.MultipleLocator(100.0/8))
	ax.yaxis.set_major_formatter(ticks)
	ax.yaxis.set_major_locator(plt.MultipleLocator(100.0/8))
