from datetime import datetime
import pandas as pd
import numpy as np
import copy
import sys
import csv
import warnings
from TreeWidthTreeContainment import BOTCH
import networkx as nx
import CPH

warnings.simplefilter(action='ignore', category=FutureWarning)

'''
Code consisting of main run file and two functions:
- run_heuristic:            1. open tree set and make CPH.PhT environment for each tree
							2. run cherry picking heuristic (CPH)
							3. return results
- run_main:                 run CPH with four "PickNextCherry" methods:
								1. ML
								2. TrivialML
								3. Rand
								4. TrivialRand

RUN in terminal:
python main_heuristic.py <instance num.> <ML model name> <leaf number> <bool (0/1) for exact input> <option>
option: 
if exact input = 0:
	option = reticulation number
else:
	option = forest size
EXAMPLE: 
python main_heuristic.py 0 N10_maxL100_random_balanced 20 0 50
'''
import gym
import torch
from gym import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from stable_baselines3 import PPO

import gym
from gym import spaces
import numpy as np


from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import dgl
import numpy as np
import pickle
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.utils import get_device

from gcnn import *

class GraphEnv(gym.Env):
	def __init__(self, network, tree_set):
		N = len(network.nw.nodes)
		self.initial_network = deepcopy(network)
		self.tree_set = deepcopy(tree_set)
		self.network = network
		self.action_space = spaces.MultiDiscrete([N,N,N,N])
		self.observation_space = gym.spaces.Box(low=0, high=1, shape=(N,N), dtype=int)

		self.num_moves = 0
		self.moves_ub = 100
		self.window = None
		self.clock = None
		self.N = N 
		self.ret_baseline = self.network.ret_number()
		print(f'{self.ret_baseline=}')
		print(self.tree_set.trees)
		print(self.network.num_trees_contained(self.tree_set.trees))

	def reset(self):
		# We need the following line to seed self.np_random
		self.network = deepcopy(self.initial_network)
		self.num_moves = 0

		observation = self._get_obs()        
		return observation

	def render(self):
		return None
	
	def _render_frame(self):
		return None

	def _get_obs(self):
		matrix = np.zeros((self.N, self.N), dtype=np.int32)
		for a,b in self.network.nw.edges:
			matrix[a,b] = 1
		return matrix

	def _get_info(self):
		return {}

	def step(self, action):
		#print('trees_contained', self.network.num_trees_contained(self.tree_set.trees), len(self.tree_set.trees))
		info = self._get_info()
		self.num_moves += 1
		if self.num_moves >= self.moves_ub:
			terminated = True
		else:
			terminated = False

		#perform move
		u,v,s,t = list(action)
		#r1 = self.network.check_edges_exist(u,v,s,t)
		#print(r1)
		self.network.tail_move(u,v, s,t)

		#calculate reward
		#for x in self.network.nx
		reward = 0
		trees_contained = self.network.num_trees_contained(self.tree_set.trees)
		
		if terminated:
			if trees_contained < len(self.tree_set.trees.values()):
				reward = trees_contained - len(self.tree_set.trees.values())
			else:
				ret_number = self.network.ret_number()
				reward = self.ret_baseline - ret_number
			print(reward)

		observation = self._get_obs()
		#print(f'{action=},{reward=}')

		return observation, reward, terminated, info


from stable_baselines3.common.env_checker import check_env
def run_search(network, tree_set):
	print('starting search')
	env = GraphEnv(network, tree_set)
	print("sample observation:", env.observation_space.sample())
	print("sample action:", env.action_space.sample())

	network.get_legal_moves()
	exit(0)
	check_env(env)
	
	model = PPO(ActorCriticGnnPolicy, env, verbose=1, tensorboard_log="my_tbdir/", policy_kwargs={"simple":True})
	model.learn(total_timesteps=10**5)


from copy import deepcopy
import random

def run_heuristic(tree_set=None, tree_set_newick=None, inst_num=0, repeats=1, time_limit=None,
				  progress=False,  reduce_trivial=False, pick_ml=False, pick_ml_triv=False,
				  pick_random=False, model_name=None, relabel=False, relabel_cher_triv=False, problem_type="",
				  full_leaf_set=True, ml_thresh=None):
	# READ TREE SET
	np.random.seed(42)
	random.seed(42)

	now = datetime.now().time()
	if progress:
		print(f"Instance {inst_num} {problem_type}: Start at {now}")

	if tree_set is None and tree_set_newick is not None:
		# Empty set of inputs
		inputs = []

		# Read each line of the input file with name set by "option_file_argument"
		f = open(tree_set_newick, "rt")
		reader = csv.reader(f, delimiter='~', quotechar='|')
		for row in reader:
			inputs.append(str(row[0]))
		f.close()

		# Make the set of inputs usable for all algorithms: use the CPH class
		tree_set = CPH.Input_Set(newick_strings=inputs, instance=inst_num, full_leaf_set=full_leaf_set)

	# RUN HEURISTIC CHERRY PICKING SEQUENCE
	# Run the heuristic to find a cherry-picking sequence `seq' for the set of input trees.
	# Arguments are set as given by the terminal arguments
	seq_dist, seq, df_pred = tree_set.CPSBound(repeats=repeats,
											   progress=progress,
											   time_limit=time_limit,
											   reduce_trivial=reduce_trivial,
											   pick_ml=pick_ml,
											   pick_ml_triv=pick_ml_triv,
											   pick_random=pick_random,
											   relabel=relabel,
											   relabel_cher_triv=relabel_cher_triv,
											   model_name=model_name,
											   ml_thresh=ml_thresh,
											   problem_type=problem_type)

	# Output the computation time for the heuristic
	now = datetime.now().time()
	if progress:
		print(f"Instance {inst_num} {problem_type}: Finish at {now}")
		print(f"Instance {inst_num} {problem_type}: Computation time heuristic: {tree_set.CPS_Compute_Time}")
		print(f"Instance {inst_num} {problem_type}: Reticulation number = {min(tree_set.RetPerTrial.values())}")
	if pick_ml:
		return tree_set.RetPerTrial, tree_set.DurationPerTrial, seq, df_pred
	else:
		return tree_set.RetPerTrial, tree_set.DurationPerTrial, seq

def run_main(i, l, exact, ret=None, forest_size=None,
			 repeats=1, time_limit=None, ml_name=None, full_leaf_set=True, ml_thresh=None, progress=False):
	if exact:
		test_info = f"L{l}_R{ret}_exact_all"
		file_info = f"L{l}_R{ret}_exact"
	else:
		test_info = f"L{l}_T{forest_size}_all"
		file_info = f"L{l}_T{forest_size}"

	# ML MODEL
	model_name = f"LearningCherries/RFModels/rf_cherries_{ml_name}.joblib"
	# save results
	score = pd.DataFrame(
		index=pd.MultiIndex.from_product([[i], ["RetNum", "Time"], np.arange(repeats)]),
		columns=["ML", "TrivialML", "Rand", "TrivialRand", "UB"], dtype=float)
	df_seq = pd.DataFrame()
	env_info_file = f"Data/Test/inst_results/tree_data_{file_info}_{i}.pickle"
	# INSTANCE
	tree_set_newick = f"Data/Test/TreeSetsNewick/tree_set_newick_{file_info}_{i}_LGT.txt"
	print(tree_set_newick)
	#########
	#read trees
	# Read each line of the input file with name set by "option_file_argument"
	inputs = []
	f = open(tree_set_newick, "rt")
	reader = csv.reader(f, delimiter='~', quotechar='|')
	for row in reader:
		inputs.append(str(row[0]))
	f.close()
	# Make the set of inputs usable for all algorithms: use the CPH class
	tree_set = CPH.Input_Set(newick_strings=inputs, instance=i, full_leaf_set=full_leaf_set)

	#run heuristic to construct initial network
	ret_score, time_score, seq_ra = run_heuristic(
		tree_set=tree_set,
		tree_set_newick=None,
		inst_num=i,
		repeats=repeats,
		time_limit=10**3,
		problem_type="Rand",
		pick_random=True,
		relabel=False,
		full_leaf_set=full_leaf_set,
		progress=progress)
	initial_network = CPH.PhN(seq=seq_ra)
	print('seq',seq_ra, initial_network.labels)
	print(tree_set.trees[0].nw.edges, tree_set.trees[0].leaves, tree_set.labels_reversed)
	#for x in tree_set.labels_reversed:
	#    tree_set.labels_reversed[x] = int(tree_set.labels_reversed[x])
	#for x in initial_network.labels:
	#    initial_network.labels[int(x)] = initial_network.labels[x]
	
	for t in tree_set.trees:
		#print(tree_set.trees[t].nw.nodes)
		nx.relabel_nodes(tree_set.trees[t].nw, tree_set.labels_reversed, copy=False)
		#print(tree_set.trees[t].nw.nodes)
		nx.relabel_nodes(tree_set.trees[t].nw, initial_network.labels, copy=False)
		#print(tree_set.trees[t].nw.nodes)
		
	for tree in tree_set.trees.values():
		#print(tree)
		print(tree.nw.edges, tree.nw.nodes)
		print(initial_network.nw.edges, initial_network.nw.nodes)
		root_nw = None
		for x in initial_network.nw.nodes:
			if initial_network.nw.in_degree[x] == 0:
				print(x)
				root_nw = x
		root_tree = None
		for x in tree.nw.nodes:
			if tree.nw.in_degree[x] == 0:
				print(x)
				root_tree = x

		T = deepcopy(tree.nw)
		N = deepcopy(initial_network.nw)
		#T.add_edge(1000, root_tree)
		#N.add_edge(1000,root_nw)
		print(N.edges, N.nodes)
		#res = BOTCH.tc_brute_force(T, N)
		res = BOTCH.tc_brute_force(T,N)
		assert res==True

	#exit(0)
	run_search(initial_network, tree_set)

if __name__ == "__main__":
	i = int(sys.argv[1])
	ml_name = str(sys.argv[2])
	l = int(sys.argv[3])
	exact_input = int(sys.argv[4])

	if exact_input:
		exact = True
		ret = int(sys.argv[5])
		forest_size = None
	else:
		exact = False
		ret = None
		forest_size = int(sys.argv[5])

	if len(sys.argv) == 7:
		ml_thresh = int(sys.argv[6])
	else:
		ml_thresh = None

	print(f'{i=}')
	print(f'{l=}')
	print(f'{exact=}')
	print(f'{ret=}')
	print(f'{forest_size=}')
	print(f'{ml_name=}')
	
	run_main(i, l, exact, ret, forest_size, ml_name=ml_name, full_leaf_set=True, ml_thresh=ml_thresh, progress=True)
