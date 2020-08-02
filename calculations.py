import numpy as np
from tqdm import trange, tqdm
import sys


def build_network(portals):
	"""

	:param portals:
	:return: discrete 3D network : Z^3 intersection with a bounded area
	"""
	shape = np.shape(portals)
	portals = portals.reshape(shape[0]*shape[1], shape[2])
	x_min, y_min, z_min = np.min(portals, axis=0)
	x_max, y_max, z_max = np.max(portals, axis=0)
	x_list = np.arange(np.trunc(x_min), np.trunc(x_max) + 1)
	y_list = np.arange(np.trunc(y_min), np.trunc(y_max) + 1)
	z_list = np.arange(np.trunc(z_min), np.trunc(z_max) + 1)
	network_shape = (len(x_list), len(y_list), len(z_list))
	x_list, y_list, z_list = np.meshgrid(x_list, y_list, z_list, indexing='ij')
	network = np.array([x_list.flatten(), y_list.flatten(), z_list.flatten()]).T
	return network, network_shape


def projection(network, portals):
	"""

	:param network:
	:param portals:
	:return: matrix of shape (len(network), len(portals)) with element i, j equal to euclidian distance between node i
	and clothest point in portal j
	"""
	portals_min, portals_max = np.transpose(portals, (1, 0, 2))
	points = np.transpose(np.array([network for _ in range(len(portals))]), (1, 0, 2))
	arg = (portals_min <= points)*(points <= portals_max)
	array = np.array([portals - node for node in network])
	argmin = np.argmin(np.abs(array), axis=2)
	argmax = np.logical_not(argmin)
	arg2 = np.transpose(np.array([argmax, argmin]), (1, 2, 0, 3))
	res = np.sum(portals*arg2, axis=2)
	res = points*arg + res*np.logical_not(arg)
	return np.linalg.norm(res - points, axis=2)


def antiprojection(network, portals):
	"""

	:param network:
	:param portals:
	:return: matrix of shape (len(network), len(portals)) with element i, j equal to euclidian distance between node i
	and furthest point in portal j
	"""
	array = np.array([portals - node for node in network])
	argmax = np.argmax(np.abs(array), axis=2)
	argmin = np.logical_not(argmax)
	arg = np.transpose(np.array([argmin, argmax]), (1, 2, 0, 3))
	points = np.sum(portals*arg, axis=2)
	network = np.transpose(np.array([network for _ in range(len(portals))]), (1, 0, 2))
	return np.linalg.norm(points - network, axis=2)


def build_region(network, network_shape, portals):
	
	regions = np.zeros((len(portals),) + network_shape, dtype=np.bool)
	proj = projection(network, portals)
	anti_proj = antiprojection(network, portals)
	proj = np.reshape(proj, network_shape + (len(portals),))
	anti_proj = np.reshape(anti_proj, network_shape + (len(portals),))
	argmin = np.argmin(proj, axis=3)
	buffer = 0

	for ind_x in trange(network_shape[0], desc="building regions", file=sys.stdout):
		for ind_y in range(network_shape[1]):
			for ind_z in range(network_shape[2]):
				am = argmin[ind_x, ind_y, ind_z]
				if anti_proj[ind_x, ind_y, ind_z, am] < np.min(np.delete(proj[ind_x, ind_y, ind_z], am)) + buffer:
					regions[am, ind_x, ind_y, ind_z] = True
	return regions


def possible_portails(network, network_shape, portals, regions, names):
	proj = projection(network, portals)
	anti_proj = antiprojection(network, portals)
	proj = np.reshape(proj, network_shape + (len(portals),))
	anti_proj = np.reshape(anti_proj, network_shape + (len(portals),))
	portal_x = np.array([1, 2, 0])
	portal_z = np.array([0, 2, 1])
	nb = len(portals)
	portals = [[] for _ in range(nb)]
	origin = network[0]
	x_shape = np.array(network_shape) - portal_x
	xl, yl, zl = np.arange(x_shape[0]), np.arange(x_shape[1]), np.arange(x_shape[2])
	xl, yl, zl = np.meshgrid(xl, yl, zl, indexing='ij')
	x_network = np.transpose([xl.flatten(), yl.flatten(), zl.flatten()])
	z_shape = np.array(network_shape) - portal_z
	xl, yl, zl = np.arange(z_shape[0]), np.arange(z_shape[1]), np.arange(z_shape[2])
	xl, yl, zl = np.meshgrid(xl, yl, zl, indexing='ij')
	z_network = np.transpose([xl.flatten(), yl.flatten(), zl.flatten()])
	for i in trange(nb, desc="finding possibles portals in regions", file=sys.stdout):
		region = regions[i]
		for n in x_network:
			fn = n + portal_x + [1, 1, 1]
			if region[n[0]:fn[0], n[1]:fn[1], n[2]:fn[2]].all():
				# if np.max(anti_proj[n[0]:fn[0], n[1]:fn[1], n[2]:fn[2], i]) < np.min(proj[n[0]:fn[0], n[1]:fn[1], n[2]:fn[2], np.delete(range(nb), i)]):
				portals[i].append(np.array([n, fn - [1, 1, 1]]) + origin)
		for n in z_network:
			fn = n + portal_z + [1, 1, 1]
			if region[n[0]:fn[0], n[1]:fn[1], n[2]:fn[2]].all():
				# if np.max(anti_proj[n[0]:fn[0], n[1]:fn[1], n[2]:fn[2], i]) < np.min(proj[n[0]:fn[0], n[1]:fn[1], n[2]:fn[2], np.delete(range(nb), i)]):
				portals[i].append(np.array([n, fn - [1, 1, 1]]) + origin)
		portals[i] = np.array(portals[i])
	return portals


def project(portals, m):
	norm = [np.linalg.norm(np.mean(portals_in_reg, axis=1) - m, axis=1) for portals_in_reg in portals]
	argmin = [np.argmin(norm_in_reg) for norm_in_reg in norm]
	portals = np.array([portals[i][argmin[i]] for i in range(len(portals))])
	return portals


def checking(res_portals, portals, names):
	nb = len(portals)
	for i in trange(nb, desc="testing result", file=sys.stdout):
		res = get_all_portal(res_portals[i])
		for res_block in res:
			dist_max = 0
			for portal_block in get_all_portal(portals[i]):
				dist = np.linalg.norm(res_block - portal_block)
				if dist_max > dist:
					dist_max = dist
			dist_min = np.inf
			for j in np.delete(range(nb), i):
				for portal_block in get_all_portal(portals[j]):
					dist = np.linalg.norm(res_block - portal_block)
					if dist_min < dist:
						dist_min = dist
			if not(dist_max < dist_min):
				raise ValueError(f"Portail {names[i]}, precisely block {res_block} is not in the proper region")
	print("no portals are in the MOJANG YOU RETARD border, and all portals are in the good region")


# def checking(res_portals, portals, names):
# 	nb = len(portals)
# 	min_dist = np.ones((nb, nb)) * np.inf
# 	max_dist = np.zeros((nb, nb))
# 	for i in trange(nb, desc="testing result", file=sys.stdout):
# 		res = get_all_portal(res_portals[i])
# 		for j in range(nb):
# 			portal = get_all_portal(portals[j])
# 			for res_block in res:
# 				for portal_block in portal:
# 					dist = np.linalg.norm(res_block - portal_block)
# 					if dist < min_dist[i, j]:
# 						min_dist[i, j] = dist
# 					if dist > max_dist[i, j]:
# 						max_dist[i, j] = dist
# 	argmin = np.argmin(min_dist, axis=1)
# 	for i in range(nb):
# 		if argmin[i] != i:
# 			raise ValueError(f"{names[i]} portal is not is the right region")
# 	print("all portals are in the good regions")
# 	for i in range(nb):
# 		if not(max_dist[i, i] < np.min(np.delete(min_dist[i], i))):
# 			print("max dist to portals = ", max_dist[i])
# 			print("min dist to portals = ", min_dist[i])
# 			print("no good NO GOOD NONONONONO :", res_portals[i])
# 			raise ValueError(f"{names[i]} portal is in the MOJANG YOU RETARD border")
# 	print("no portals are in the MOJANG YOU RETARD border \n  -------------------- ")


def get_all_portal(portal):
	delta = portal[1] - portal[0]
	if delta[0] == 0:
		return np.array([portal[0], portal[0] + [0, 0, delta[2]], portal[0] + [0, 1, 0], portal[0] + [0, 1, delta[2]], portal[0] + [0, 2, 0], portal[1]])
	else:
		return np.array([portal[0], portal[0] + [delta[0], 0, 0], portal[0] + [0, 1, 0], portal[0] + [delta[0], 1, 0], portal[0] + [0, 2, 0], portal[1]])
