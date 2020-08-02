import numpy as np
from calculations import Network, build_region, project, possible_portails


def find_portals(ovw_portals, portals_names):
	"""

	:param ovw_portals: overworld portals positions
	:param portals_names: overworld portals names
	:return: optimal portals positions in the nether
	"""
	m_0 = np.mean(ovw_portals, axis=(0, 1))
	m_0 = np.array([m_0[1]/8, m_0[1], m_0[2]])
	print(f"initial average mass of portals = {m_0} \n --------------------")

	print(f"building network")
	network = Network(ovw_portals)
	print("network built \n --------------------")

	regions = build_region(network, ovw_portals)
	print("--------------------")

	pos_portail = possible_portails(network, regions)
	print(" --------------------")

	m_new = m_0
	print("starting iterative projection of the mass center")
	portals_found = None
	for j in range(666):
		portals_found = project(pos_portail, m_new)
		m_new = np.mean(portals_found, axis=(0, 1))
		if (m_new == m_0).all():
			print("converged in ", j+1, " itérations")
			break
		m_0 = m_new
	else:
		print(f"did not converged in {666} iterations")
	print(f"solution center = {m_new} \n  --------------------")
	return portals_found
