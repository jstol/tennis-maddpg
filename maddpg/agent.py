#!/usr/bin/env python
# coding: utf-8
"""Agent definition."""
# TODO this class/module should probably also eventually handle generating actions and updating the agent.

# Standard imports
from typing import List

# Third party imports
from torch import optim

# Project imports
from .models import QNet, PolicyNet
from .utils import copy_weights, device


class Agent:
    def __init__(self, state_size: int = 24, action_size: int = 2, hidden_sizes: List[int] = None,
                 lr_policy: float = 1e-3, lr_q: float = 1e-2, weight_decay: float = 0.0, num_env_agents: int = 2,
                 use_batchnorm_policy: bool = True, use_batchnorm_q: bool = True):
        """DDPG Agent.

        Args:
            state_size: Size of the observation space.
            action_size: Size of the action space.
            hidden_sizes: The number of neurons to use within each hidden layer.
            lr_policy: The Policy ("Actor") learning rate.
            lr_q: The Q-Net ("Critic") learning rate.
            weight_decay: The strength of L2 regularization to use.
            num_env_agents: The number of agents in the environment.
            use_batchnorm_policy: If set to True, enables batchnorm in Policy (Actor).
            use_batchnorm_q: If set to True, enables batchnorm in Q-Net (Critic).
        """

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes or [256, 128]

        # "Actor" and "Actor" target network
        self.policy_net = PolicyNet(state_size=state_size, action_size=action_size,
                                    hidden_sizes=hidden_sizes, use_batchnorm=use_batchnorm_policy).to(device)
        self.policy_net_target = PolicyNet(state_size=state_size, action_size=action_size,
                                           hidden_sizes=hidden_sizes, use_batchnorm=use_batchnorm_policy).to(device)
        copy_weights(self.policy_net, self.policy_net_target, 1.0)

        # "Critic" and "Critic" target network
        self.q_net = QNet(state_size=state_size * num_env_agents, action_size=action_size * num_env_agents,
                          hidden_sizes=hidden_sizes, use_batchnorm=use_batchnorm_q).to(device)
        self.q_net_target = QNet(state_size=state_size * num_env_agents, action_size=action_size * num_env_agents,
                                 hidden_sizes=hidden_sizes, use_batchnorm=use_batchnorm_q).to(device)
        copy_weights(self.q_net, self.q_net_target, 1.0)

        # Enable nets for training (for batchnorm)
        self.policy_net.train()
        self.q_net.train()

        # Set up training params and optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_policy, weight_decay=weight_decay)
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=lr_q, weight_decay=weight_decay)
