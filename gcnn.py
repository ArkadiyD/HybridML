import os.path as osp
import numpy as np
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, Batch
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

from feature_extractors import *
from utils import *
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.distributions import MaskableDistribution, make_masked_proba_distribution
import torch as th

class MyGraphDataset(InMemoryDataset):
	def __init__(self, graph, transform=None):
		super(MyGraphDataset, self).__init__('.', transform, None, None)
		self.data, self.slices = self.collate(graph)


class ActorCriticGnnPolicy(MaskableActorCriticPolicy):
	def __init__(
			self,
			observation_space: gym.spaces.Space,
			action_space: gym.spaces.Space,
			lr_schedule: Schedule,
			net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
			activation_fn: Type[nn.Module] = nn.Tanh,
			simple: bool = False,
			legal_checker = None,
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

		super(MaskableActorCriticPolicy, self).__init__(
			observation_space,
			action_space,
			features_extractor_class,
			features_extractor_kwargs,
			optimizer_class=optimizer_class,
			optimizer_kwargs=optimizer_kwargs,
			squash_output=squash_output,
		)
		self.simple = simple
		self.legal_checker = legal_checker
		print(f'{simple=}')
		print(observation_space)
		self.N = observation_space['graph'].shape[0]
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

		self.features_extractor = features_extractor_class(
			self.observation_space, **self.features_extractor_kwargs)
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
			warnings.warn(
				"sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

		self.use_sde = use_sde
		self.dist_kwargs = dist_kwargs

		# Action distribution
		self.action_dist = make_masked_proba_distribution(action_space)

		self._build(lr_schedule)

	def _get_constructor_parameters(self) -> Dict[str, Any]:
		data = super()._get_constructor_parameters()

		default_none_kwargs = self.dist_kwargs or collections.defaultdict(
			lambda: None)

		data.update(
			dict(
				net_arch=self.net_arch,
				activation_fn=self.activation_fn,
				use_sde=self.use_sde,
				log_std_init=self.log_std_init,
				squash_output=default_none_kwargs["squash_output"],
				full_std=default_none_kwargs["full_std"],
				use_expln=default_none_kwargs["use_expln"],
				# dummy lr schedule, not needed for loading policy alone
				lr_schedule=self._dummy_schedule,
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
		assert isinstance(
			self.action_dist, StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
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
			self.gnn_extractor = GNNSimpleExtractor(self.N*2+4).cuda()

	def _build(self, lr_schedule: Schedule) -> None:
		"""
		Create the networks and the optimizer.
		:param lr_schedule: Learning rate schedule
				lr_schedule(1) is the initial learning rate
		"""
		self._build_gnn_extractor()

		latent_dim_pi = self.gnn_extractor.latent_dim_pi

		self.action_net = self.action_dist.proba_distribution_net(latent_dim=self.gnn_extractor.latent_dim_pi)
		self.value_net = nn.Linear(self.gnn_extractor.latent_dim_vf, 1)
		self.aux_net = nn.Linear(self.gnn_extractor.latent_dim_pi, 1)
		
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
		self.optimizer = self.optimizer_class(
			self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

		loss = torch.nn.BCELoss()
		# for k in range(1000):
		#	x = np.random.randint()
		print(self.value_net)
		print(self.action_net)
		# exit(0)

	def extract_features(self, obs):
		if self.simple:
			return self.extract_features_simple(obs)
		else:
			return self.extract_features_gnn(obs)

	def extract_features_gnn(self, obs):
		graph = obs['graph']
		picked_nodes = obs['picked_nodes'].long()
		
		batch_size = graph.shape[0]
		all_data = []
		for k in range(batch_size):
			edge_index = []
			#for i in range(graph.shape[1]):
			#	for j in range(graph.shape[2]):
			#		if graph[k, i, j] == 1:
			#			edge_index.append((i, j))
			#ind = torch.nonzero(graph[k])
			#print(ind, ind.shape)

			edge_index = torch.nonzero(graph[k]).long()
			# print(edge_index.shape)
			edge_index = edge_index.t().contiguous()
			# print(edge_index.shape)
			cur_features = torch.zeros((graph[k].shape[0], 4))
			for i, picked in enumerate(picked_nodes[k]):
				#print(i,picked,cur_features.shape)
				cur_features[picked.item(), i] = 1
			#print(cur_features)

			all_data.append(Data(x=torch.tensor(cur_features).float(), edge_index=edge_index))
		features = Batch.from_data_list(data_list = all_data).cuda()
		return features

	def extract_features_simple(self, obs):
		graph = obs['graph']
		picked_nodes = obs['picked_nodes']

		batch_size = graph.shape[0]
		all_features = []
		for k in range(batch_size):
			in_degree = [0 for x in range(graph[k].shape[0])]
			out_degree = [0 for x in range(graph[k].shape[0])]

			for i in range(graph[k].shape[0]):
				for j in range(graph[k].shape[1]):
					if graph[k, i, j] == 1:
						in_degree[j] += 1
						out_degree[i] += 1
			in_degree = np.array(in_degree)
			out_degree = np.array(out_degree)
			features = np.vstack(
				[in_degree.reshape(1, -1), out_degree.reshape(1, -1)]).reshape(-1)
			all_features.append(features)
		all_features = torch.tensor(all_features).cuda().float()
		picked_nodes = torch.tensor(picked_nodes).cuda().float()
		
		all_features = torch.cat((all_features, picked_nodes),dim=1)
		#print(all_features.shape, picked_nodes.shape)

		return all_features

	def forward(
		self,
		obs: th.Tensor,
		deterministic: bool = False,
		action_masks: Optional[np.ndarray] = None,
	) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
		"""
		Forward pass in all the networks (actor and critic)
		:param obs: Observation
		:param deterministic: Whether to sample or use deterministic actions
		:param action_masks: Action masks to apply to the action distribution
		:return: action, value and log probability of the action
		"""
		# Preprocess the observation if needed
		print(f'{action_masks=}')
		print('picked_nodes', obs['picked_nodes'])

		features = self.extract_features(obs)
		latent_pi, latent_vf = self.gnn_extractor(features)
		# Evaluate the values for the given observations
		values = self.value_net(latent_vf)
		distribution = self._get_action_dist_from_latent(latent_pi)
		if action_masks is not None:
			distribution.apply_masking(action_masks)
		actions = distribution.get_actions(deterministic=deterministic)
		log_prob = distribution.log_prob(actions)
		return actions, values, log_prob

	def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> MaskableDistribution:
		"""
		Retrieve action distribution given the latent codes.
		:param latent_pi: Latent code for the actor
		:return: Action distribution
		"""
		action_logits = self.action_net(latent_pi)
		return self.action_dist.proba_distribution(action_logits=action_logits)

	def _predict(
		self,
		observation: th.Tensor,
		deterministic: bool = False,
		action_masks: Optional[np.ndarray] = None,
	) -> th.Tensor:
		"""
		Get the action according to the policy for a given observation.
		:param observation:
		:param deterministic: Whether to use stochastic or deterministic actions
		:param action_masks: Action masks to apply to the action distribution
		:return: Taken action according to the policy
		"""
		return self.get_distribution(observation, action_masks).get_actions(deterministic=deterministic)

	def evaluate_actions(
		self,
		obs: th.Tensor,
		actions: th.Tensor,
		action_masks: Optional[np.ndarray] = None,
	) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
		"""
		Evaluate actions according to the current policy,
		given the observations.
		:param obs: Observation
		:param actions: Actions
		:return: estimated value, log likelihood of taking those actions
			and entropy of the action distribution.
		"""
		print('evaluate')
		features = self.extract_features(obs)
		latent_pi, latent_vf = self.gnn_extractor(features)

		distribution = self._get_action_dist_from_latent(latent_pi)
		if action_masks is not None:
			distribution.apply_masking(action_masks)
		print('evaluate',action_masks.shape)
		log_prob = distribution.log_prob(actions)
		values = self.value_net(latent_vf)
		return values, log_prob, distribution.entropy()

	def get_distribution(self, obs: th.Tensor, action_masks: Optional[np.ndarray] = None) -> MaskableDistribution:
		"""
		Get the current policy distribution given the observations.
		:param obs: Observation
		:param action_masks: Actions' mask
		:return: the action distribution.
		"""
		features = super().extract_features(obs, self.pi_features_extractor)
		latent_pi = self.mlp_extractor.forward_actor(features)
		distribution = self._get_action_dist_from_latent(latent_pi)
		if action_masks is not None:
			distribution.apply_masking(action_masks)
		return 

	def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
		"""
		Get the estimated values according to the current policy given the observations.
		:param obs:
		:return: the estimated values.
		"""
		features = self.extract_features(obs)
		latent_vf = self.gnn_extractor.forward_critic(features)
		print('values', latent_vf.shape, latent_vf)
		return self.value_net(latent_vf)


