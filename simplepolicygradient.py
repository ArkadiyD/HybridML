import sys
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union

import gym
import numpy as np
import torch as th
from feature_extractors import *
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data import DataLoader, Batch
from torch_geometric.utils import degree
from torch_geometric.nn import GINConv, global_add_pool

class Perceptron(nn.Module):
	def __init__(self, num_features, out_classes, device='cuda'):
		super(Perceptron, self).__init__()
		self.layers = [64,64]
		self.seq = []
		prev_f = num_features
		for l in self.layers:
			self.seq.append(torch.nn.Linear(prev_f, l))
			prev_f = l
			self.seq.append(torch.nn.ReLU())
		self.seq.append(torch.nn.Linear(prev_f, out_classes))		
		self.seq = torch.nn.Sequential(*self.seq)

	def forward(self, x):
		return self.seq(x)

class SimplePolicyGradient:
	def __init__(self, env, mask_function=None):
		self.env = env
		self.obs_space = env.observation_space
		self.action_space = env.action_space
		self.n_actions = self.action_space.n
		self.mask_function = mask_function
		print(self.n_actions)

		self.N = self.env.observation_space['graph'].shape[0]
		#self.model = GCN(self.N, 4, self.n_actions).cuda()
		self.model = GAT(self.N, 4, self.n_actions).cuda()
		
		#self.model = Perceptron(self.N*self.N + self.N*4, self.n_actions).cuda()
		
		self.learning_rate = 1e-4
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
		self.Horizon = 32
		self.MAX_TRAJECTORIES = 100000
		self.gamma = 1.0

		from torch.utils.tensorboard import SummaryWriter
		self.writer = SummaryWriter()

		print(self.model)

	def extract_features(self, obs):
		if isinstance(self.model, GIN) or isinstance(self.model, GCN) or isinstance(self.model, GAT):
			return self.extract_features_gnn(obs)
		elif isinstance(self.model, Perceptron):
			return self.extract_features_simple(obs)
		else:
			raise NotImplementedError()

	def extract_features_batch(self, obs):
		if isinstance(self.model, GIN) or isinstance(self.model, GCN) or isinstance(self.model, GAT):
			return self.extract_features_gnn_batch(obs)
		elif isinstance(self.model, Perceptron):
			return self.extract_features_simple_batch(obs)
		else:
			raise NotImplementedError()
	
	def extract_features_simple(self, obs):
		graph = torch.tensor(obs['graph']).long()
		picked_nodes = torch.tensor(obs['picked_nodes']).long()

		all_features = []
		
		#in_degree = torch.zeros((graph.shape[0],))
		#out_degree = torch.zeros((graph.shape[0],))
		#print(graph.shape)
		cur_features = torch.zeros((graph.shape[0], 4))
		
		for i, picked in enumerate(picked_nodes):
			if picked.item() >= 0:
				cur_features[picked.item(), i] = 1
		cur_features = cur_features.view(-1)
		
		graph = graph.view(-1)		
		all_features = torch.cat([graph, cur_features]).view(1,-1)
		#print(all_features.shape)
		all_features -= 0.5
		return all_features.cuda()


	def extract_features_simple_batch(self, obs):
		all_data = []
		#print(obs)
		for k in range(len(obs)):
			graph = torch.tensor(obs[k]['graph']).long()
			picked_nodes = torch.tensor(obs[k]['picked_nodes']).long()

			all_features = []
			
			#in_degree = torch.zeros((graph.shape[0],))
			#out_degree = torch.zeros((graph.shape[0],))
			#print(graph.shape)
			cur_features = torch.zeros((graph.shape[0], 4))
			
			for i, picked in enumerate(picked_nodes):
				#print(picked)
				if picked.item() >= 0:
					cur_features[picked.item(), i] = 1
			cur_features = cur_features.view(-1)
			
			graph = graph.view(-1)		
			all_features = torch.cat([graph, cur_features]).view(1,-1)
			#print(all_features.shape)
			all_data.append(all_features)
		all_data = torch.cat(all_data, dim=0)-0.5
		#print(all_data.shape)

		return all_data.cuda()

	def extract_features_gnn(self, obs):
		graph = torch.tensor(obs['graph']).long()
		picked_nodes = torch.tensor(obs['picked_nodes']).long()
		#print(graph.shape, picked_nodes.shape)

		all_data = []
		edge_index = []
		
		edge_index = torch.nonzero(graph).long()
		edge_index = edge_index.t().contiguous()
		cur_features = torch.zeros((graph.shape[0], 4))
		for i, picked in enumerate(picked_nodes):
			if picked.item() >= 0:
				cur_features[picked.item(), i] = 1
		all_data.append(Data(x=torch.tensor(cur_features).float(), edge_index=edge_index))
		features = Batch.from_data_list(data_list = all_data).cuda()
		cur_features = cur_features.view(-1)
		return (features, cur_features)
	
	def extract_features_gnn_batch(self, obs):
		all_data = []
		all_features = []
		#print('obs',len(obs))
		for k in range(len(obs)):
			#print(k)
			graph = torch.tensor(obs[k]['graph']).long()
			picked_nodes = torch.tensor(obs[k]['picked_nodes']).long()
			#print(graph.shape, picked_nodes.shape)

			edge_index = []
			
			edge_index = torch.nonzero(graph).long()
			edge_index = edge_index.t().contiguous()
			cur_features = torch.zeros((graph.shape[0], 4))
			for i, picked in enumerate(picked_nodes):
				if picked.item() >= 0:
					cur_features[picked.item(), i] = 1
			all_data.append(Data(x=torch.tensor(cur_features).float(), edge_index=edge_index))
			all_features.append(cur_features.view(1,-1))

		#print(len(all_data))
		features = Batch.from_data_list(data_list = all_data).cuda()
		all_features = torch.cat(all_features).view(len(all_data),-1).cuda()
		return (features, all_features)

	def learn(self):
		self.score = []
		batch_loss = 0.0
		
		for trajectory in range(self.MAX_TRAJECTORIES):
			curr_state = self.env.reset()
			done = False
			transitions = [] 
			
			for t in range(self.Horizon):
				all_features = self.extract_features(curr_state)#.cuda()
				#print(all_features[0], all_features[1])
				features, aux_features = all_features[0].cuda(), all_features[1].cuda()
				logits = self.model((features, aux_features))

				if self.mask_function is not None:
					legal_actions = torch.tensor(self.mask_function(self.env)).cuda().view(logits.shape)
					#print(legal_actions)
					#print(legal_actions.shape, logits.shape, logits)
					if torch.sum(legal_actions) > 0:
						logits = logits * legal_actions
					#print(logits.shape)
					act_prob = torch.softmax(logits, dim=1).view(-1)
					#print('after softmax',act_prob)
					act_prob = torch.clamp(act_prob, 0, 1)
				
				else:
					act_prob = torch.softmax(logits, dim=1).view(-1)
					act_prob = torch.clamp(act_prob, 0, 1)
				#print(act_prob)
				action = torch.multinomial(act_prob, 1).item()
				#print(action)
				prev_state = curr_state
				curr_state, reward, done, info = self.env.step(action) 
				transitions.append((prev_state, action, reward)) 
				if done: 
					break
			self.score.append(len(transitions))
			reward_batch = torch.Tensor([r for (s,a,r) in 
						transitions])#.flip(dims=(0,)) 
			#print(reward_batch)
			batch_Gvals =[]
			for i in range(len(transitions)):
				new_Gval=0
				power=0
				for j in range(i,len(transitions)):
					new_Gval=new_Gval+((self.gamma**power)*reward_batch[j]).numpy()
					power+=1
					#print(i,j,reward_batch[j], self.gamma**power, power, new_Gval)
				batch_Gvals.append(new_Gval)
			print(reward_batch, batch_Gvals)
			reward_batch = torch.cumsum(reward_batch.view(-1), dim=0)
			expected_returns_batch=torch.FloatTensor(reward_batch).cuda()
			#expected_returns_batch /= torch.max(torch.abs(expected_returns_batch))
			#expected_returns_batch -= expected_returns_batch.mean()
			#print([s[0] for s in transitions])
			state_batch = self.extract_features_batch([s[0] for s in transitions]) 
			#print(len(state_batch))
			action_batch = torch.Tensor([a[1] for a in transitions]).cuda()
			#print(state_batch)
			pred_batch = torch.softmax(self.model(state_batch), dim=1)
			#print(pred_batch.shape, torch.sum(pred_batch[0]))
			prob_batch = pred_batch.gather(dim=1,index=action_batch.long().view(-1,1)).squeeze() 
			prob_batch = torch.clamp(prob_batch, 1e-6, 1-1e-6)
			#print(prob_batch)
			#print(expected_returns_batch)
			loss= -torch.mean(torch.log(prob_batch)*expected_returns_batch) 
			print(loss.item())
			reward_batch = np.array(reward_batch).flatten()
			print(reward_batch)

			batch_loss += loss/32.0

			if trajectory % 32 == 0:
				self.optimizer.zero_grad()
				batch_loss.backward()
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
				self.optimizer.step()
				#self.logger.record("train/entropy_loss", np.mean(entropy_losses))
				
				self.writer.add_scalar("train/loss", batch_loss.item(), trajectory)
				batch_loss = 0.0

			#self.writer.add_histogram("reward", values=torch.tensor(reward_batch), global_step=trajectory, bins=len(set(reward_batch)))
			self.writer.add_scalar("episode/avg_reward", np.mean(reward_batch), trajectory)
			self.writer.add_scalar("episode/max_reward", np.max(reward_batch), trajectory)
			self.writer.add_scalar("episode/legal_actions", np.where(reward_batch >= 0)[0].shape[0] / float(len(reward_batch)), trajectory)
			self.writer.add_scalar("episode/seq_len", len(reward_batch), trajectory)
