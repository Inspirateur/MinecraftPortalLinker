import numpy as np
from tqdm import trange, tqdm
import sys


class Network:
	def __init__(self, ovw_portals):
		"""

		:param ovw_portals: data of the overworld portals, shape (2,3)
		:return: network and usefull data about network
		the network is a discrete 3D network : Z^3 intersection with a bounded area in the overworld of shape (net_shape, 3)
		"""
		shape = np.shape(ovw_portals)
		portals = ovw_portals.reshape(shape[0] * shape[1], shape[2])
		x_min, y_min, z_min = np.min(portals, axis=0)
		x_max, y_max, z_max = np.max(portals, axis=0)
		x = np.arange(x_min - 16, x_max + 16 + 1)  # Ce 16 A OPTIMISER ET NE PAS HARDCODER
		y = np.arange(y_min, y_max + 16 + 1)
		z = np.arange(z_min - 16, z_max + 16 + 1)
		self.ovw_shape = (len(x), len(y), len(z))
		x, y, z = np.meshgrid(x, y, z, indexing='ij')
		self.ovw_network = np.reshape(np.array([x.flatten(), y.flatten(), z.flatten()]).T, self.ovw_shape + (3,))
		self.ovw_origin = self.ovw_network[0, 0, 0]
		x = np.arange(np.floor(x_min/8), np.floor(x_max/8) + 1)
		y = np.arange(y_min, y_max + 1)
		z = np.arange(np.floor(z_min/8), np.floor(z_max/8) + 1)
		self.neth_shape = (len(x), len(y), len(z))
		x, y, z = np.meshgrid(x, y, z, indexing='ij')
		self.neth_network = np.reshape(np.array([x.flatten(), y.flatten(), z.flatten()]).T, self.neth_shape + (3,))
		self.neth_origin = self.ovw_network[0, 0, 0]


def projection(network, ovw_portals):
	"""

	:param network: discrete 3D network : Z^3 intersection with a bounded area in the overworld of shape (net_shape, 3)
	:param ovw_portals: data of the overworld portals, shape (number of portals, 2,3)
	:return: float tensor of shape (number of portals, net_shape) with i th element containing the euclidian distance
	between network and i th portal
	"""
	print(np.shape(network.ovw_network))
	ovw_portals = ovw_portals + 0.5  # placing portals blocks at the middle of the blocks
	net_length = network.ovw_shape[0]*network.ovw_shape[1]*network.ovw_shape[2]
	ovw_network = np.reshape(network.ovw_network, (net_length, 3))
	portals_min, portals_max = np.transpose(ovw_portals, (1, 0, 2))
	print("saucisse")
	points = np.transpose(np.array([ovw_network for _ in range(len(ovw_portals))]), (1, 0, 2))
	print("saucisse1")
	portals_arg = (portals_min <= points)*(points <= portals_max)
	print("saucisse2")
	array = np.array([ovw_portals - node for node in ovw_network])
	print("saucisse3")
	argmin = np.argmin(np.abs(array), axis=2)
	argmax = np.logical_not(argmin)
	arg2 = np.transpose(np.array([argmax, argmin]), (1, 2, 0, 3))
	res = np.sum(ovw_portals*arg2, axis=2)
	res = points*portals_arg + res*np.logical_not(portals_arg)
	proj = np.transpose(np.reshape(np.linalg.norm(res - points, axis=2), network.ovw_shape + (len(ovw_portals),)), (3, 0, 1, 2))
	return proj


def build_region(network, ovw_portals):
	regions = np.zeros((len(ovw_portals),) + network.ovw_shape, dtype=np.bool)
	proj = projection(network, ovw_portals)
	argmin = np.argmin(proj, axis=0)
	for ix in trange(network.ovw_shape[0], desc="building regions", file=sys.stdout):
		for iy in range(network.ovw_shape[1]):
			for iz in range(network.ovw_shape[2]):
				am = argmin[ix, iy, iz]
				regions[am, ix, iy, iz] = True
	return regions


def possible_portails(network, regions):
	pos_portail = [[] for _ in range(len(regions))]
	for i in trange(len(regions), desc="finding possibles portals in regions", file=sys.stdout):
		for ix in range(network.neth_shape[0]):
			for iy in range(network.neth_shape[1]):
				for iz in range(network.neth_shape[2]):
					y = iy*8 - network.ovw_origin[1]
					x_min = ix*8 - network.ovw_origin[0]
					x_max = (ix + 2)*8 - network.ovw_origin[0]  # CES COORDONNES DOIVENT ETRE OPTIMISABLES, VOIR DANS MINECRAFT
					z_min = np.floor((iz - 0.3)*8).astype(int) - network.ovw_origin[2]
					z_max = np.floor((iz + 1.3)*8).astype(int) - network.ovw_origin[2]
					if regions[i, x_min:x_max + 1, y, z_min:z_max + 1].all():
						pos_portail[i].append(np.array([[ix, iy, iz], [ix + 1, iy + 2, iz]]))
					x_min = np.floor((ix - 0.3)*8).astype(int) - network.ovw_origin[0]
					x_max = np.floor((ix + 1.3)*8).astype(int) - network.ovw_origin[0]
					z_min = iz*8 - network.ovw_origin[2]
					z_max = (iz + 2)*8 - network.ovw_origin[2]
					if regions[x_min:x_max + 1, y, z_min:z_max + 1].all():
						pos_portail[i].append(np.array([[ix, iy, iz], [ix, iy + 2, iz + 1]]))
	return pos_portail


def project(pos_portail, m):
	norm = [np.linalg.norm(np.mean(portals_in_reg, axis=1) - m, axis=1) for portals_in_reg in pos_portail]
	argmin = [np.argmin(norm_in_reg) for norm_in_reg in norm]
	portals_found = np.array([pos_portail[i][argmin[i]] for i in range(len(pos_portail))])
	return portals_found
