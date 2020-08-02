import yaml
import numpy as np
from solver import find_portals
from loss import mean_square
from time import time


with open("overworld_portals.yaml", "r") as fp:
	portals = yaml.load(fp)
ovw_portals = np.array([p for p in portals.values()])
start = time()
neth_portals = find_portals(ovw_portals, list(portals.keys()))
delta = time() - start
maxlen = max(len(name) for name in portals.keys()) + 1
print("optimal nether portals : \n")
for name, p in zip(portals.keys(), neth_portals):
	print(f"{(name+':').ljust(maxlen)} {p[0]} --> {p[1]}")
print(f"\n loss {mean_square(neth_portals):.1f} (in {int(delta*1000)} ms)")
