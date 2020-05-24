import numpy as np


def solve(vpoint):
	points = vpoint.points
	av = np.mean(points, axis=0)
	x_min, z_min = np.min(points, axis=0)
	x_max, z_max = np.max(points, axis=0)
	network = build_network(x_min, z_min, x_max, z_max)
	n_points = len(points)
	region = [[] for _ in range(n_points)]
	for node in network:
		nearest_p = np.int(np.argmin(np.linalg.norm(points - node, axis=1)))
		region[nearest_p].append(node)
	print("average =", av)
	res = [[] for _ in range(n_points)]
	for i in range(n_points):
		region_i = np.array(region[i])
		res[i] = region_i[np.int(np.argmin(np.linalg.norm(region_i - av, axis=1)))]
	return np.array(res)


def build_network(x_min, z_min, x_max, z_max):
	x_list = np.arange(np.trunc(x_min) - 1, x_max + 1)
	z_list = np.arange(np.trunc(z_min) - 0.5, z_max + 1)
	x_list, z_list = np.meshgrid(x_list, z_list)
	netowrk_1 = np.array([x_list.flatten(), z_list.flatten()]).T
	x_list = np.arange(np.trunc(x_min) - 0.5, x_max + 1)
	z_list = np.arange(np.trunc(z_min) - 1, z_max + 1)
	x_list, z_list = np.meshgrid(x_list, z_list)
	netowrk_2 = np.array([x_list.flatten(), z_list.flatten()]).T
	netowrk = np.concatenate([netowrk_1, netowrk_2])
	return netowrk
