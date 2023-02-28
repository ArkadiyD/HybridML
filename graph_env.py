import random
from copy import deepcopy
from gcnn import *
import torch.nn as nn
import pickle
import dgl
from gym import spaces
import torch
import gym
from datetime import datetime
import numpy as np
import copy
import sys
import csv

from TreeWidthTreeContainment import BOTCH
import networkx as nx
import CPH
from utils import *


class GraphEnv(gym.Env):
	def __init__(self, network, tree_set):
		N = len(network.nw.nodes)
		self.initial_network = deepcopy(network)
		self.tree_set = deepcopy(tree_set)
		self.network = network
		self.action_space = spaces.Discrete(N)
		self.observation_space = gym.spaces.Dict({
			'graph':gym.spaces.Box(low=0, high=1, shape=(N, N), dtype=int),
			'picked_nodes': gym.spaces.Box(low=-1, high=N-1, shape=(4,), dtype=int)
		})

		self.num_moves = 0
		self.moves_ub = 100
		self.window = None
		self.clock = None
		self.N = N
		self.ret_baseline = self.network.ret_number()
		print(f'{self.ret_baseline=}')
		print(self.tree_set.trees)
		print('num_trees_contained',
			  self.network.num_trees_contained(self.tree_set.trees))
		print(self.network.nw.nodes)
		self.currently_picked_nodes = []

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
		for a, b in self.network.nw.edges:
			matrix[a, b] = 1

		picked_nodes = deepcopy(self.currently_picked_nodes)
		while len(picked_nodes) < 4:
			picked_nodes.append(-1)

		return {'graph':matrix, 'picked_nodes':np.array(picked_nodes)}

	def obs(self):
		return self._get_obs()
		
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

		# perform move
		reward = 0
		
		x = action
		self.currently_picked_nodes.append(x)
		#print(f'{self.currently_picked_nodes=}')

		if len(self.currently_picked_nodes) < 4:
			is_legal = check_if_move_is_legal(self.network.nw, self.currently_picked_nodes)
			#print(is_legal)
			if not is_legal:
				reward = -2
				self.currently_picked_nodes = []

		elif len(self.currently_picked_nodes) == 4:
			u, v, s, t = list(self.currently_picked_nodes)
			is_legal = check_if_move_is_legal(self.network.nw, self.currently_picked_nodes)
			
			if not is_legal:
				reward = -2
			else:
				self.network.tail_move(u, v, s, t)

				# calculate reward
				# for x in self.network.nx
				reward = 0
				trees_contained = self.network.num_trees_contained(
					self.tree_set.trees)
				n_trees = len(self.tree_set.trees.values())
				if trees_contained < n_trees:
					# 0(all contained) : 1 (no contained)
					reward = (n_trees - trees_contained) / float(n_trees)
					reward = -reward  # -1 (no contained) : 0 (all_contained)
				else:
					ret_number = self.network.ret_number()
					reward = 1.0 - ret_number / \
						float(self.ret_baseline)  # [0;1] #lower is better
				# print(reward)
			self.currently_picked_nodes = []

		reward = float(reward)
		observation = self._get_obs()
		#print(reward)
		

		return observation, reward, terminated, info


class GraphEnvLearnLegalMoves(GraphEnv):
	def step(self, action):
		#print('trees_contained', self.network.num_trees_contained(self.tree_set.trees), len(self.tree_set.trees))
		
		info = self._get_info()
		self.num_moves += 1
		if self.num_moves >= self.moves_ub:
			terminated = True
		else:
			terminated = False

		# perform move
		reward = 0
		
		x = action
		self.currently_picked_nodes.append(x)
		
		if len(self.currently_picked_nodes) < 4:
			is_legal = check_if_move_is_legal(self.network.nw, self.currently_picked_nodes)
			if not is_legal:
				reward = -1
				self.currently_picked_nodes = []
			else:
				reward = 1
				
		elif len(self.currently_picked_nodes) == 4:
			u, v, s, t = list(self.currently_picked_nodes)
			is_legal = check_if_move_is_legal(self.network.nw, self.currently_picked_nodes)
			
			if not is_legal:
				reward = -1
			else:
				self.network.tail_move(u, v, s, t)

				# calculate reward
				# for x in self.network.nx
				reward = 10
				# trees_contained = self.network.num_trees_contained(
				# 	self.tree_set.trees)
				# n_trees = len(self.tree_set.trees.values())
				# if trees_contained < n_trees:
				# 	# 0(all contained) : 1 (no contained)
				# 	reward = (n_trees - trees_contained) / float(n_trees)
				# 	reward = -reward+1.1  # -0.1 (no contained) : 1.1 (all_contained)
				# else:
				# 	ret_number = self.network.ret_number()
				# 	reward = 2.2 - ret_number / \
				# 		float(self.ret_baseline)  # [1.2;2] #lower is better
				# print(reward)
				terminated = True

			self.currently_picked_nodes = []
			
		#if reward <  0:
		#	terminated = True

		reward = float(reward) #+ len(self.currently_picked_nodes)
		observation = self._get_obs()
		print(reward, self.currently_picked_nodes)
		

		return observation, reward, terminated, info

