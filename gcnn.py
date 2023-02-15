import os.path as osp
import numpy as np
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
from torch_geometric.nn import GINConv, global_add_pool
from sklearn.model_selection import StratifiedKFold
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Tuple
from torch_geometric.data import Data
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor
from stable_baselines3.common.distributions import (
	BernoulliDistribution,
	CategoricalDistribution,
	DiagGaussianDistribution,
	Distribution,
	MultiCategoricalDistribution,
	StateDependentNoiseDistribution,
	make_proba_distribution,
)
from stable_baselines3.common.torch_layers import (
	BaseFeaturesExtractor,
	CombinedExtractor,
	FlattenExtractor,
	MlpExtractor,
	NatureCNN,
	create_mlp,
)
import gym
from typing import *
from functools import partial
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data


class MyGraphDataset(InMemoryDataset):
	def __init__(self, graph, transform=None):
		super(MyGraphDataset, self).__init__('.', transform, None, None)
		data = graph
		self.data, self.slices = self.collate([data])
		
class ActorCriticGnnPolicy(ActorCriticPolicy):
	def __init__(
		self,
		observation_space: gym.spaces.Space,
		action_space: gym.spaces.Space,
		lr_schedule: Schedule,
		net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
		activation_fn: Type[nn.Module] = nn.Tanh,
		simple: bool = False,
		ortho_init: bool = True,
		use_sde: bool = False,
		log_std_init: float = 0.0,
		full_std: bool = True,
		sde_net_arch: Optional[List[int]] = None,
		use_expln: bool = False,
		squash_output: bool = False,
		features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
		features_extractor_kwargs: Optional[Dict[str, Any]] = None,
		normalize_images: bool = True,
		optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
		optimizer_kwargs: Optional[Dict[str, Any]] = None,
	):

		if optimizer_kwargs is None:
			optimizer_kwargs = {}
			# Small values to avoid NaN in Adam optimizer
			if optimizer_class == torch.optim.Adam:
				optimizer_kwargs["eps"] = 1e-5

		super(ActorCriticPolicy, self).__init__(
			observation_space,
			action_space,
			features_extractor_class,
			features_extractor_kwargs,
			optimizer_class=optimizer_class,
			optimizer_kwargs=optimizer_kwargs,
			squash_output=squash_output,
		)
		self.simple = simple
		print(f'{simple=}')
		self.N = observation_space.shape[0]
		print(f'{self.N=}')
		# Default network architecture, from stable-baselines
		if net_arch is None:
			if features_extractor_class == NatureCNN:
				net_arch = []
			else:
				net_arch = [dict(pi=[64, 64], vf=[64, 64])]

		self.net_arch = net_arch
		self.activation_fn = activation_fn
		self.ortho_init = ortho_init

		self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
		self.features_dim = self.features_extractor.features_dim

		self.normalize_images = normalize_images
		self.log_std_init = log_std_init
		dist_kwargs = None
		# Keyword arguments for gSDE distribution
		if use_sde:
			dist_kwargs = {
				"full_std": full_std,
				"squash_output": squash_output,
				"use_expln": use_expln,
				"learn_features": False,
			}

		if sde_net_arch is not None:
			warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

		self.use_sde = use_sde
		self.dist_kwargs = dist_kwargs

		# Action distribution
		self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

		self._build(lr_schedule)

	def _get_constructor_parameters(self) -> Dict[str, Any]:
		data = super()._get_constructor_parameters()

		default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

		data.update(
			dict(
				net_arch=self.net_arch,
				activation_fn=self.activation_fn,
				use_sde=self.use_sde,
				log_std_init=self.log_std_init,
				squash_output=default_none_kwargs["squash_output"],
				full_std=default_none_kwargs["full_std"],
				use_expln=default_none_kwargs["use_expln"],
				lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
				ortho_init=self.ortho_init,
				optimizer_class=self.optimizer_class,
				optimizer_kwargs=self.optimizer_kwargs,
				features_extractor_class=self.features_extractor_class,
				features_extractor_kwargs=self.features_extractor_kwargs,
			)
		)
		return data

	def reset_noise(self, n_envs: int = 1) -> None:
		"""
		Sample new weights for the exploration matrix.
		:param n_envs:
		"""
		assert isinstance(self.action_dist, StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
		self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

	def _build_gnn_extractor(self) -> None:
		"""
		Create the policy and value networks.
		Part of the layers can be shared.
		"""
		# Note: If net_arch is None and some features extractor is used,
		#       net_arch here is an empty list and mlp_extractor does not
		#       really contain any layers (acts like an identity module).
		if not self.simple:
			self.gnn_extractor = GNNExtractor().cuda()
		else:
			self.gnn_extractor = GNNSimpleExtractor(self.N*2).cuda()

	def _build(self, lr_schedule: Schedule) -> None:
		"""
		Create the networks and the optimizer.
		:param lr_schedule: Learning rate schedule
			lr_schedule(1) is the initial learning rate
		"""
		self._build_gnn_extractor()

		latent_dim_pi = self.gnn_extractor.latent_dim_pi

		if isinstance(self.action_dist, DiagGaussianDistribution):
			self.action_net, self.log_std = self.action_dist.proba_distribution_net(
				latent_dim=latent_dim_pi, log_std_init=self.log_std_init
			)
		elif isinstance(self.action_dist, StateDependentNoiseDistribution):
			self.action_net, self.log_std = self.action_dist.proba_distribution_net(
				latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
			)
		elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
			self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
		else:
			raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

		self.value_net = nn.Linear(self.gnn_extractor.latent_dim_vf, 1)
		# Init weights: use orthogonal initialization
		# with small initial weight for the output
		if self.ortho_init:
			# TODO: check for features_extractor
			# Values from stable-baselines.
			# features_extractor/mlp values are
			# originally from openai/baselines (default gains/init_scales).
			module_gains = {
				self.features_extractor: np.sqrt(2),
				self.gnn_extractor: np.sqrt(2),
				self.action_net: 0.01,
				self.value_net: 1,
			}
			for module, gain in module_gains.items():
				module.apply(partial(self.init_weights, gain=gain))

		# Setup optimizer with initial learning rate
		self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

	def extract_features(self, obs):
		if self.simple:
			return self.extract_features_simple(obs)
		else:
			return self.extract_features_gnn(obs)

	def extract_features_gnn(self, obs):
		obs = obs[0]
		edge_index = []
		for i in range(obs.shape[0]):
			for j in range(obs.shape[1]):
				if obs[i,j] == 1:
					edge_index.append((i,j))
		edge_index = torch.tensor(edge_index, dtype=torch.long)
		#print(edge_index.shape)
		edge_index = edge_index.t().contiguous()
		#print(edge_index.shape)
		data = Data(x = torch.tensor([[1] for i in range(obs.shape[0])], dtype=torch.float), edge_index=edge_index)
		dataset = MyGraphDataset(data)
		#print(dataset)
		loader = DataLoader(dataset, batch_size=1)
		for features in loader:
			features = features.cuda()
			#print(features.is_undirected())
			break
		return features

	def extract_features_simple(self, obs):
		obs = obs[0]
		in_degree = [0 for x in range(obs.shape[0])]
		out_degree = [0 for x in range(obs.shape[0])]

		for i in range(obs.shape[0]):
			for j in range(obs.shape[1]):
				if obs[i,j] == 1:
					in_degree[j] += 1
					out_degree[i] += 1
		in_degree = np.array(in_degree)
		out_degree = np.array(out_degree)
		features = np.vstack([in_degree.reshape(1,-1), out_degree.reshape(1,-1)]).flatten()
		features = torch.tensor(features).cuda().float().reshape(1,-1)
		return features

	def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Forward pass in all the networks (actor and critic)
		:param obs: Observation
		:param deterministic: Whether to sample or use deterministic actions
		:return: action, value and log probability of the action
		"""
		# Preprocess the observation if needed
		#print('observation', obs.shape)
		features = self.extract_features(obs)
		#print(features)
		latent_pi, latent_vf = self.gnn_extractor(features)
		# Evaluate the values for the given observations
		values = self.value_net(latent_vf)
		#print(f'{values=}')
		distribution = self._get_action_dist_from_latent(latent_pi)
		actions = distribution.get_actions(deterministic=deterministic)
		#print(f'{actions=}')
		log_prob = distribution.log_prob(actions)
		#print(log_prob)
		return actions, values, log_prob

	def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
		"""
		Retrieve action distribution given the latent codes.
		:param latent_pi: Latent code for the actor
		:return: Action distribution
		"""
		mean_actions = self.action_net(latent_pi)
		#print(mean_actions)
		if isinstance(self.action_dist, DiagGaussianDistribution):
			return self.action_dist.proba_distribution(mean_actions, self.log_std)
		elif isinstance(self.action_dist, CategoricalDistribution):
			# Here mean_actions are the logits before the softmax
			return self.action_dist.proba_distribution(action_logits=mean_actions)
		elif isinstance(self.action_dist, MultiCategoricalDistribution):
			# Here mean_actions are the flattened logits
			return self.action_dist.proba_distribution(action_logits=mean_actions)
		elif isinstance(self.action_dist, BernoulliDistribution):
			# Here mean_actions are the logits (before rounding to get the binary actions)
			return self.action_dist.proba_distribution(action_logits=mean_actions)
		elif isinstance(self.action_dist, StateDependentNoiseDistribution):
			return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
		else:
			raise ValueError("Invalid action distribution")

	def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
		"""
		Get the action according to the policy for a given observation.
		:param observation:
		:param deterministic: Whether to use stochastic or deterministic actions
		:return: Taken action according to the policy
		"""
		return self.get_distribution(observation).get_actions(deterministic=deterministic)

	def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Evaluate actions according to the current policy,
		given the observations.
		:param obs:
		:param actions:
		:return: estimated value, log likelihood of taking those actions
			and entropy of the action distribution.
		"""
		# Preprocess the observation if needed
		features = self.extract_features(obs)
		latent_pi, latent_vf = self.gnn_extractor(features)
		distribution = self._get_action_dist_from_latent(latent_pi)
		log_prob = distribution.log_prob(actions)
		values = self.value_net(latent_vf)
		return values, log_prob, distribution.entropy()

	def get_distribution(self, obs: torch.Tensor) -> Distribution:
		"""
		Get the current policy distribution given the observations.
		:param obs:
		:return: the action distribution.
		"""
		features = self.extract_features(obs)
		latent_pi = self.gnn_extractor.forward_actor(features)
		return self._get_action_dist_from_latent(latent_pi)

	def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
		"""
		Get the estimated values according to the current policy given the observations.
		:param obs:
		:return: the estimated values.
		"""
		features = self.extract_features(obs)
		latent_vf = self.gnn_extractor.forward_critic(features)
		return self.value_net(latent_vf)

class GNNExtractor(nn.Module):
	def __init__(
	self,
#	feature_dim: int,
#	net_arch: List[Union[int, Dict[str, List[int]]]],
#	activation_fn: Type[nn.Module],
#	device: Union[torch.device, str] = "auto"):
	device='cuda'
	):
		super(GNNExtractor, self).__init__()
		device = get_device(device)
		shared_net, policy_net, value_net = [], [], []
		policy_only_layers = [64,64]  # Layer sizes of the network that only belongs to the policy network
		value_only_layers = [64,64]  # Layer sizes of the network that only belongs to the value network
		last_layer_dim_shared = 10

		self.shared_net = GIN()
		print(self.shared_net)

		last_layer_dim_pi = last_layer_dim_shared
		last_layer_dim_vf = last_layer_dim_shared
		activation_fn = torch.nn.ReLU()
		
		# Build the non-shared part of the network
		for pi_layer_size, vf_layer_size in zip(policy_only_layers, value_only_layers):
			if pi_layer_size is not None:
				assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
				policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
				policy_net.append(activation_fn)
				last_layer_dim_pi = pi_layer_size

			if vf_layer_size is not None:
				assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
				value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
				value_net.append(activation_fn)
				last_layer_dim_vf = vf_layer_size

		# Save dim, used to create the distributions
		self.latent_dim_pi = last_layer_dim_pi
		self.latent_dim_vf = last_layer_dim_vf

		# Create networks
		# If the list of layers is empty, the network will just act as an Identity module
		self.policy_net = nn.Sequential(*policy_net).to(device)
		self.value_net = nn.Sequential(*value_net).to(device)
		print(self.policy_net)
		print(self.value_net)

	def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		:return: latent_policy, latent_value of the specified network.
			If all layers are shared, then ``latent_policy == latent_value``
		"""
		#print(f'{features=}')
		shared_latent = self.shared_net(features)
		#print(f'{shared_latent=}')
		return self.policy_net(shared_latent), self.value_net(shared_latent)

	def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
		return self.policy_net(self.shared_net(features))

	def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
		return self.value_net(self.shared_net(features))

class GNNSimpleExtractor(nn.Module):
	def __init__(
	self,
#	feature_dim: int,
#	net_arch: List[Union[int, Dict[str, List[int]]]],
#	activation_fn: Type[nn.Module],
#	device: Union[torch.device, str] = "auto"):
	num_features,
	device='cuda'
	):
		super(GNNSimpleExtractor, self).__init__()
		device = get_device(device)
		shared_net, policy_net, value_net = [], [], []
		policy_only_layers = [64,64]  # Layer sizes of the network that only belongs to the policy network
		value_only_layers = [64,64]  # Layer sizes of the network that only belongs to the value network
		shared_layers = [64,64]  # Layer sizes of the network that only belongs to the value network

		last_layer_dim_shared = 10

		shared_net =[]
		print(shared_net)
		last_layer_dim_shared = num_features
		activation_fn = torch.nn.ReLU()
		
		for i, shared_layer_size in enumerate(shared_layers):
			if shared_layer_size is not None:
				assert isinstance(shared_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
				shared_net.append(nn.Linear(last_layer_dim_shared, shared_layer_size))
				shared_net.append(activation_fn)
				last_layer_dim_shared = shared_layer_size

		last_layer_dim_pi = last_layer_dim_shared
		last_layer_dim_vf = last_layer_dim_shared

		# Build the non-shared part of the network
		for pi_layer_size, vf_layer_size in zip(policy_only_layers, value_only_layers):
			if pi_layer_size is not None:
				assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
				policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
				policy_net.append(activation_fn)
				last_layer_dim_pi = pi_layer_size

			if vf_layer_size is not None:
				assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
				value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
				value_net.append(activation_fn)
				last_layer_dim_vf = vf_layer_size

		# Save dim, used to create the distributions
		self.latent_dim_pi = last_layer_dim_pi
		self.latent_dim_vf = last_layer_dim_vf

		# Create networks
		# If the list of layers is empty, the network will just act as an Identity module
		self.shared_net = nn.Sequential(*shared_net).to(device)		
		self.policy_net = nn.Sequential(*policy_net).to(device)
		self.value_net = nn.Sequential(*value_net).to(device)
		print(self.shared_net)		
		print(self.policy_net)
		print(self.value_net)

	def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		:return: latent_policy, latent_value of the specified network.
			If all layers are shared, then ``latent_policy == latent_value``
		"""
		#print(f'{features=}', features.shape)
		shared_latent = self.shared_net(features)
		#print(f'{shared_latent=}')
		return self.policy_net(shared_latent), self.value_net(shared_latent)

	def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
		return self.policy_net(self.shared_net(features))

	def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
		return self.value_net(self.shared_net(features))


class GIN(nn.Module):
		def __init__(self):
			super(GIN, self).__init__()

			num_features = 1#dataset.num_features
			self.dim = 64
			self.latent_dim = 10
			self.dropout = 0.5

			self.num_layers = 4

			self.convs = nn.ModuleList()
			self.bns = nn.ModuleList()
			self.fcs = nn.ModuleList()

			self.convs.append(GINConv(nn.Sequential(nn.Linear(num_features, self.dim), nn.BatchNorm1d(self.dim), nn.ReLU(), nn.Linear(self.dim, self.dim))))
			self.bns.append(nn.BatchNorm1d(self.dim))
			self.fcs.append(nn.Linear(num_features, self.latent_dim))
			self.fcs.append(nn.Linear(self.dim, self.latent_dim))

			for i in range(self.num_layers-1):
				self.convs.append(GINConv(nn.Sequential(nn.Linear(self.dim, self.dim), nn.BatchNorm1d(self.dim), nn.ReLU(), nn.Linear(self.dim, self.dim))))
				self.bns.append(nn.BatchNorm1d(self.dim))
				self.fcs.append(nn.Linear(self.dim, self.latent_dim))
		
		def reset_parameters(self):
			for m in self.modules():
				if isinstance(m, nn.Linear):
					m.reset_parameters()
				elif isinstance(m, GINConv):
					m.reset_parameters()
				elif isinstance(m, nn.BatchNorm1d):
					m.reset_parameters()

		def forward(self, data):
			#if len(data.shape) == 2:
			#	data = torch.unsqueeze(data,0)
			#print(data.shape)

			x = data.x
			edge_index = data.edge_index
			batch = data.batch
			#print(x, edge_index, batch)

			outs = [x]
			for i in range(self.num_layers):
				#print(i, x.shape)
				x = self.convs[i](x, edge_index)
				x = self.bns[i](x)
				x = F.relu(x)
				outs.append(x)
				#print('outs',i,x.shape, len(outs))

			#for k in outs:
			#	print(k.shape)
			#print(len(outs), outs[0].shape)

			out = None
			for i, x in enumerate(outs):
				#print(i,x.shape)
				x = global_add_pool(x, batch)
				#print(x.shape)
				x = F.dropout(self.fcs[i](x), p=self.dropout, training=self.training)
				if out is None:
					out = x
				else:
					out += x
			#print('out', out.shape)
			return out
			#return F.log_softmax(out, dim=-1), 0
