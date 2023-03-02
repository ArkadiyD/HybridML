import random
from copy import deepcopy
from gcnn import *
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
import torch.nn as nn
import pickle
import dgl
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from gym import spaces
import torch
import gym
from datetime import datetime
import pandas as pd
import numpy as np
import copy
import sys
import csv
import warnings

from myppo import PPOWithAuxLoss
from mymaskableppo import MyMaskablePPO
from TreeWidthTreeContainment import BOTCH
import networkx as nx
import CPH
from utils import *
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from simplepolicygradient import SimplePolicyGradient
from graph_env import GraphEnv, GraphEnvLearnLegalMoves

from NetworkGen.NetworkToTree import *
from NetworkGen.LGT_network import *
from NetworkGen.tree_to_newick import *

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

def mask_fn(env: gym.Env) -> np.ndarray:
	# Do whatever you'd like in this function to return the action mask
	# for the current env. In this example, we assume the env has a
	# helpful method we can rely on.
	cur_picked_nodes = env.currently_picked_nodes
	#print(cur_picked_nodes)
	#nodes = [1 for i in env.network.nw.nodes]
	#print(nodes)
	nodes = get_next_legal_nodes(env.network.nw, cur_picked_nodes)
	#print(nodes)
	return nodes

def run_search(network, tree_set):
	print('starting search')
	env = GraphEnv(network, tree_set)
	env = ActionMasker(env, mask_fn)

	# for k in range(15):
	# 	#print("sample observation:", env.observation_space.sample())
	# #net
	# 	legal_moves = get_next_legal_nodes(env.network.nw, env.obs()['picked_nodes'])
	# 	legal_moves = legal_moves.astype(np.int32)
	# 	print(legal_moves)
	# 	N = env.get_obs()['graph'].shape[0]
	# 	probs = np.array([1/float(N) for x in range(N)])
	# 	if np.sum(legal_moves) > 0:
	# 		probs[legal_moves==0] = 0
	# 		probs /= np.sum(probs)
	# 	print(env._get_obs()['picked_nodes'])
	# 	print(probs)
	# 	action = torch.multinomial(torch.tensor(probs).float(), 1).item()
	# 	print("sample action:", action)
	# 	_,reward,_,_ = env.step(action)
	# 	print(f'{reward=}')
	get_legal_moves(network.nw)
	check_env(env)
	#exit(0)
	model = MyMaskablePPO(ActorCriticGnnPolicy, env, verbose=1,
						   tensorboard_log="my_tbdir/", policy_kwargs={"simple": False, "legal_checker":get_next_legal_nodes})
	model.learn(total_timesteps=10**5)
	#model = SimplePolicyGradient(env, mask_function=None)
	#model.learn()

def generate_random_network(ret, l):
	beta = 1
	distances = True

	# make network
	n = l - 2 + ret
	print(f"JOB: Start creating NETWORK (In-Sample, L = {l}, R = {ret}, n = {n})")
	while True:
		if l <= 20:
			alpha = np.random.uniform(0.1, 0.5)
		elif l <= 50:
			alpha = np.random.uniform(0.1, 0.3)
		else:
			alpha = np.random.uniform(0.1, 0.2)
		net, ret_num = simulation(n, alpha, 1, beta)
		num_leaves = len(leaves(net))
		if num_leaves == l and ret_num == ret:
			break
		else:
			print(f"JOB: NETWORK GEN FAILED, again (In-Sample, L = {l}, R = {ret}, n = {n})")
	return net

def run_heuristic(tree_set=None, tree_set_newick=None, inst_num=0, repeats=1, time_limit=None,
				  progress=False,  reduce_trivial=False, pick_ml=False, pick_ml_triv=False,
				  pick_random=False, model_name=None, relabel=False, relabel_cher_triv=False, problem_type="",
				  full_leaf_set=True, ml_thresh=None):
	# READ TREE SET
	np.random.seed(43)
	random.seed(43)

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
		tree_set = CPH.Input_Set(
			newick_strings=inputs, instance=inst_num, full_leaf_set=full_leaf_set)

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
		print(
			f"Instance {inst_num} {problem_type}: Computation time heuristic: {tree_set.CPS_Compute_Time}")
		print(
			f"Instance {inst_num} {problem_type}: Reticulation number = {min(tree_set.RetPerTrial.values())}")
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
		index=pd.MultiIndex.from_product(
			[[i], ["RetNum", "Time"], np.arange(repeats)]),
		columns=["ML", "TrivialML", "Rand", "TrivialRand", "UB"], dtype=float)
	df_seq = pd.DataFrame()
	env_info_file = f"Data/Test/inst_results/tree_data_{file_info}_{i}.pickle"
	# INSTANCE
	tree_set_newick = f"Data/Test/TreeSetsNewick/tree_set_newick_{file_info}_{i}_LGT.txt"
	print(tree_set_newick)
	#########
	# read trees
	# Read each line of the input file with name set by "option_file_argument"
	inputs = []
	f = open(tree_set_newick, "rt")
	reader = csv.reader(f, delimiter='~', quotechar='|')
	for row in reader:
		inputs.append(str(row[0]))
	f.close()
	# Make the set of inputs usable for all algorithms: use the CPH class
	# run heuristic to construct initial network
	tree_set = CPH.Input_Set(newick_strings=inputs,
							 instance=i, full_leaf_set=full_leaf_set)

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
	print(seq_ra)
	#exit(0)

	#ret_score = np.min(list(ret_score.values()))-1
	initial_network = generate_random_network(ret+2, l)
	
	#for x in initial_network.nodes:
	#	if initial_network.out_degree[x] == 0:
	#		print('leave', x)
	#		nx.relabel_nodes(tree_set.trees[t].nw, lambda x:int(x), copy=False)

	#print(initial_network.edges)
	#print(initial_network)
	
	# network = CPH.PhN(seq=seq_ra)
	# # print('seq', seq_ra, initial_network.labels)
	# # print(tree_set.trees[0].nw.edges,
	# # 	  tree_set.trees[0].leaves, tree_set.labels_reversed)
	# print(network.nw.nodes, network.labels)
	# exit(0)

	rename_leaves = {}
	i = 0
	network_leaves = [x for x in initial_network.nodes if initial_network.out_degree(x) == 0]
	for x in tree_set.trees:
		cnt = 0
		#print(len(tree_set.trees[x].nw.nodes), tree_set.trees[x].nw.nodes)
		for leaf in network_leaves:
			if leaf in tree_set.trees[x].nw.nodes:
				nx.relabel_nodes(tree_set.trees[x].nw, {leaf:np.max(tree_set.trees[x].nw.nodes)+1}, copy=False) #rename nodes in the tree which are in the leaves set of the network

		tree_leaves_ = [i for i in tree_set.trees[x].nw.nodes if tree_set.trees[x].nw.out_degree(i) == 0]
		not_leaves_ = [i for i in tree_set.trees[x].nw.nodes if tree_set.trees[x].nw.out_degree(i) > 0]
		
		print(len(tree_leaves_), tree_leaves_, not_leaves_, set(not_leaves_).intersection(set(network_leaves)))
		
		for leaf in tree_leaves_: #rename leaves in the tree to match the leaves set of the network
			nx.relabel_nodes(tree_set.trees[x].nw, {leaf:network_leaves[cnt]}, copy=False)
			cnt += 1
			print([i for i in tree_set.trees[x].nw.nodes if tree_set.trees[x].nw.out_degree(i) == 0])
		
		tree_leaves_ = [i for i in tree_set.trees[x].nw.nodes if tree_set.trees[x].nw.out_degree(i) == 0]
		assert sorted(tree_leaves_) == sorted(network_leaves)

	tree_leaves = [x for x in tree_set.trees[0].nw.nodes if tree_set.trees[0].nw.out_degree(x) == 0]

	print(initial_network.out_degree)
	print(tree_leaves, network_leaves)
	print(initial_network.nodes)
	
	assert min(initial_network.nodes) == 0
	assert max(initial_network.nodes) == len(initial_network.nodes)-1
	
	#exit(0)
	initial_network = PhN(net=initial_network)
	initial_network.print_network()		
	exit(0)
	
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

	run_main(i, l, exact, ret, forest_size, ml_name=ml_name,
			 full_leaf_set=True, ml_thresh=ml_thresh, progress=True)
