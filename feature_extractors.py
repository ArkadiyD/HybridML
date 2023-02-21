from torch import nn
import torch
from typing import Tuple
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
from torch_geometric.nn import GINConv, global_add_pool
import torch.nn.functional as F


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
        # Layer sizes of the network that only belongs to the policy network
        policy_only_layers = [64]
        # Layer sizes of the network that only belongs to the value network
        value_only_layers = [64]
        last_layer_dim_shared = 64

        self.shared_net = GIN(num_features=4)
        print(self.shared_net)

        last_layer_dim_pi = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared
        activation_fn = torch.nn.ReLU()

        # Build the non-shared part of the network
        for pi_layer_size, vf_layer_size in zip(policy_only_layers, value_only_layers):
            if pi_layer_size is not None:
                assert isinstance(
                    pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                policy_net.append(activation_fn)
                last_layer_dim_pi = pi_layer_size

            if vf_layer_size is not None:
                assert isinstance(
                    vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                value_net.append(activation_fn)
                last_layer_dim_vf = vf_layer_size
        #value_net.append(nn.Linear(last_layer_dim_vf, 1))
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
        # print(f'{shared_latent=}')
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
        # Layer sizes of the network that only belongs to the policy network
        policy_only_layers = [64]
        # Layer sizes of the network that only belongs to the value network
        value_only_layers = [64]
        # Layer sizes of the network that only belongs to the value network
        shared_layers = [64, 64]

        shared_net = []
        print(shared_net)
        last_layer_dim_shared = num_features
        activation_fn = torch.nn.ReLU()

        for i, shared_layer_size in enumerate(shared_layers):
            if shared_layer_size is not None:
                assert isinstance(
                    shared_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                shared_net.append(
                    nn.Linear(last_layer_dim_shared, shared_layer_size))
                shared_net.append(activation_fn)
                last_layer_dim_shared = shared_layer_size

        last_layer_dim_pi = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared

        # Build the non-shared part of the network
        for pi_layer_size, vf_layer_size in zip(policy_only_layers, value_only_layers):
            if pi_layer_size is not None:
                assert isinstance(
                    pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                policy_net.append(activation_fn)
                last_layer_dim_pi = pi_layer_size

            if vf_layer_size is not None:
                assert isinstance(
                    vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                value_net.append(activation_fn)
                last_layer_dim_vf = vf_layer_size
        #value_net.append(nn.Linear(last_layer_dim_vf, 1))
        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.shared_net = nn.Sequential(*shared_net).to(device)
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)
        self.aux_head = nn.Linear(last_layer_dim_shared, 1)
        print(self.shared_net)
        print(self.policy_net)
        print(self.value_net)
        print(self.aux_head)
        #exit(0)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
                If all layers are shared, then ``latent_policy == latent_value``
        """
        #print(f'{features=}', features.shape)
        shared_latent = self.shared_net(features)
        # print(f'{shared_latent=}')
        return self.policy_net(shared_latent), self.value_net(shared_latent)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(self.shared_net(features))

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(self.shared_net(features))

    def forward_aux(self, features: torch.Tensor) -> torch.Tensor:
        return self.aux_head(self.shared_net(features))


class GIN(nn.Module):
    def __init__(self, num_features, out_layer = None):
        super(GIN, self).__init__()

        #num_features = num_features  # dataset.num_features
        self.dim = 64
        self.latent_dim = 64
        self.dropout = 0.5

        self.num_layers = 2

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        self.convs.append(GINConv(nn.Sequential(nn.Linear(num_features, self.dim), nn.BatchNorm1d(
            self.dim), nn.ReLU(), nn.Linear(self.dim, self.dim))))
        self.bns.append(nn.BatchNorm1d(self.dim))
        self.fcs.append(nn.Linear(num_features, self.latent_dim))
        self.fcs.append(nn.Linear(self.dim, self.latent_dim))

        for i in range(self.num_layers-1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(self.dim, self.dim), nn.BatchNorm1d(
                self.dim), nn.ReLU(), nn.Linear(self.dim, self.dim))))
            self.bns.append(nn.BatchNorm1d(self.dim))
            self.fcs.append(nn.Linear(self.dim, self.latent_dim))

        self.out_layer = out_layer
        if self.out_layer is not None:
            self.final_layer = nn.Linear(self.latent_dim, self.out_layer)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, GINConv):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, data):
        # if len(data.shape) == 2:
        #	data = torch.unsqueeze(data,0)
        # print(data.shape)

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

        # for k in outs:
        #	print(k.shape)
        #print(len(outs), outs[0].shape)

        out = None
        for i, x in enumerate(outs):
            # print(i,x.shape)
            x = global_add_pool(x, batch)
            # print(x.shape)
            x = F.dropout(self.fcs[i](x), p=self.dropout,
                          training=self.training)
            if out is None:
                out = x
            else:
                out += x
        #print('out', out.shape)
        if self.out_layer is not None:
            out = self.final_layer(out)
        return out
        # return F.log_softmax(out, dim=-1), 0
