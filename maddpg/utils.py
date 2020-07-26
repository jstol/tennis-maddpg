#!/usr/bin/env python
# coding: utf-8
"""Various helpers (for managing the environment and training)."""

# Standard imports
from typing import (
    Generator,
    List,
    Tuple,
)

# Third party imports
import numpy as np
from unityagents import UnityEnvironment
from torch import nn
import torch


# Define model/environment helpers
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def env_step(env: UnityEnvironment, brain_name: str, action: np.array) -> Tuple[np.array, float, bool]:
    """Helper function to wrap Unity env.step

    Args:
        env: An instance of the environment.
        brain_name: The name of the Unity "brain" to use.
        action: The action(s) that have been selected.

    Returns:
        A tuple of the state transitioned to, the reward received, and whether or not the episode has finished.
    """
    env_info = env.step(action)[brain_name]
    state = env_info.vector_observations
    reward = env_info.rewards
    done = any(env_info.local_done)
    return state, reward, done


def env_reset(env: UnityEnvironment, brain_name: str, train_mode: bool = True) -> List[float]:
    """Helper function to reset Unity env

    Args:
        env: An instance of the environment.
        brain_name: The name of the Unity "brain" to use.
        train_mode: If True, reset the env in training mode.

    Returns:
        The starting state.
    """
    env_info = env.reset(train_mode=train_mode)[brain_name]
    return env_info.vector_observations


# Training-specific helpers
def process_action(action: np.array, noise_stdev: float = 0.1, min_action_val: float = -1.0,
                   max_action_val: float = 1.0, add_noise: bool = True) -> np.array:
    """Process an action before feeding to env by adding (Gaussian) noise and clipping to acceptable bounds.

    Note: See the following paper for more info on this approach versus OU noise:
        > Fujimoto, Scott, Herke van Hoof and David Meger. “Addressing Function Approximation
        > Error in Actor-Critic Methods.” ArXiv abs/1802.09477 (2018): n. pag.

    Args:
        action: The action to process.
        noise_stdev: The standard deviation to use when sampling from (Gaussian) noise distribution.
        min_action_val: The minimum value to clip to.
        max_action_val: The maximum value to clip to.
        add_noise: If True, add noise to the action (before clipping).

    Returns:
        The processed action.
    """
    noise_sample = np.random.normal(scale=noise_stdev, size=action.shape) if add_noise else 0.0
    return np.clip(action + noise_sample, min_action_val, max_action_val)


def noise_stdev_generator(starting_stdev: float = 0.1, min_stdev: float = 1e-3, decay_rate: float = 0.999) \
        -> Generator[float, None, None]:
    """Generates a decaying standard deviation.

     To help DDPG explore at the beginning of training. Should be fed into a Gaussian noise generator (to introduce
     noise into the action space).

    Args:
        starting_stdev: The standard deviation to start at.
        min_stdev: The minimum standard deviation to decay to.
        decay_rate: The rate of decay.

    Returns:
        A generator that yields the standard deviation (to be called per episode during training).
    """
    stdev = starting_stdev
    while True:
        yield stdev
        stdev = max(stdev * decay_rate, min_stdev)


def copy_weights(net: nn.Module, target_net: nn.Module, tau: float = 0.001):
    """Update target_dqn model parameters.

    (Function based on Udacity DeepRL DQN homework code).
    θ_target_dqn = [τ * θ_dqn] + [(1 - τ) * θ_target_dqn]

    Args:
        net: The model to copy params from.
        target_net: The model to mix params into.
        tau: The mixing coefficient (what percentage of the source weights to use).
    """
    for net_params, target_net_params in zip(net.parameters(), target_net.parameters()):
        target_net_params.data.copy_(tau * net_params.data + (1.0 - tau) * target_net_params.data)
