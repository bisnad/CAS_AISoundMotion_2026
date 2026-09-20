import numpy as np
import json

class NPZ_Tools():

	@staticmethod
	def load(motion_filename, topology_filename):
		npz_data = NPZ_Tools.load_npz(motion_filename)
		topo_data = NPZ_Tools.load_topology(topology_filename)
		return npz_data, topo_data

	@staticmethod
	def load_npz(motion_filename):
	    with np.load(motion_filename) as data:
	        motion = dict(data)
	    return motion

	@staticmethod
	def build_children_from_parents(parents):
		children = [[] for _ in range(len(parents))]
		for j_idx, p_idx in enumerate(parents):
			if p_idx >= 0:
				children[p_idx].append(j_idx)
		return children

	@staticmethod
	def load_topology(topology_filename):
		with open(topology_filename, "r") as f:
			topo = json.load(f)
		parents = topo["jointParents"]
		children = topo.get("jointChildren", NPZ_Tools.build_children_from_parents(parents))
		joints = topo.get("jointNames", [f"joint_{j}" for j in range(len(parents))])
		return parents, children, joints
