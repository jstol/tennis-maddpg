#!/usr/bin/env python
# coding: utf-8
"""Functions to train/visualize models."""

# Standard imports
import os
import random
from collections import defaultdict, deque
from typing import (
    List,
)

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from unityagents import UnityEnvironment

# Project imports
from .agent import Agent
from .models import PolicyNet
from .utils import device, env_reset, env_step, process_action, noise_stdev_generator, copy_weights

# Define a global path to the Unity environment (can be parameterized later, if needed)
APP_PATH = 'Tennis.app'


# Replay Buffer
class ReplayBuffer:
    """Stores (S, A, R, S') tuples observed during training."""
    def __init__(self, maxlen: int = 1e6):
        # Arbitrarily keep a max of 1M tuples by default (could be tuned in the future)
        self._maxlen = maxlen
        self._buffer = deque(maxlen=self._maxlen)

    def add(self, state: List[float], action: List[float], reward: float, next_state: List[float], done: bool):
        # Store (s, a, r, s', done) pairs
        self._buffer.append((state, action, reward, next_state, int(done)))

    def get_batch(self, batch_size) -> tuple:
        sample = random.sample(self._buffer, k=batch_size)
        states, actions, rewards, next_states, dones = zip(*sample)

        return (
            Tensor(states).float(),
            Tensor(actions).float(),
            Tensor(rewards).float().unsqueeze(-1),
            Tensor(next_states).float(),
            Tensor(dones).float().unsqueeze(-1),
        )

    def __len__(self):
        return len(self._buffer)


# Define main functions
def train_model():
    """Trains an MADDPG model."""
    # TODO These could be passed in as arguments, if desired
    # Constants for General setup
    SCORE_WINDOW_SIZE = 100         # The window size to use when calculating the running average
    AVERAGE_SCORE_GOAL = 0.5        # The running average goal (determines if the environment has been solved)

    # Hyperparams
    RANDOM_EXPLORATION_SPAN = 150   # The number of episodes to do random exploration for before using policy
    MAX_EPISODES = int(5e3)         # The maximum number of episodes to run for
    BATCH_SIZE = 256                # The minibatch size to use during SGD
    UPDATE_FREQ = 10                # How frequently (in terms of steps) the models should be updated
    NUM_INNER_UPDATE = 10           # When updating the models, how many SGD steps to run
    NOISE_STDEV = 0.25              # The (initial) standard deviation to use when sampling noise (Normal dist.)
    NOISE_STDEV_DECAY = 0.998       # The amount to decay the noise stdev by each episode
    MAX_REPLAY_SIZE = int(1e5)      # The maximum number of SARS' tuples to store during training (FIFO)
    GAMMA = 0.99                    # The future reward discount factor
    TAU = 1e-3                      # The (soft) mixing factor to use when updating target network weights
    LR_POLICY = 2e-3                # The learning rate to use for the policy net ("Actor")
    LR_Q = 2e-3                     # The learning rate to use for the Q net ("Critic")
    WEIGHT_DECAY = 0.0              # The strength of L2 regularization
    HIDDEN_SIZES = [512, 256]       # The size/number of hidden layers to use (for all networks)
    USE_BATCHNORM_ACTOR = True      # If set to true, will use batch normalization (Actor)
    USE_BATCHNORM_CRITIC = False    # If set to true, will use batch normalization (Actor)

    # Set up environment
    env = UnityEnvironment(file_name=APP_PATH)
    brain_name = env.brain_names[0]
    _env_info = env.reset(train_mode=True)[brain_name]  # Note: this is a throwaway reset simply to get info on the env
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    state_size = _env_info.vector_observations.shape[1]
    num_agents = len(_env_info.agents)
    noise_stdev_gen = noise_stdev_generator(starting_stdev=NOISE_STDEV, decay_rate=NOISE_STDEV_DECAY)

    # Create agents
    agents = []
    for _ in range(num_agents):
        agent = Agent(state_size=state_size, action_size=action_size, hidden_sizes=HIDDEN_SIZES, lr_policy=LR_POLICY,
                      lr_q=LR_Q, weight_decay=WEIGHT_DECAY, num_env_agents=num_agents,
                      use_batchnorm_policy=USE_BATCHNORM_ACTOR, use_batchnorm_q=USE_BATCHNORM_CRITIC)
        agents.append(agent)

    # Set up required data structures (replay buffer and episode score trackers)
    replay_buffer = ReplayBuffer(MAX_REPLAY_SIZE)
    scores = []
    scores_window = deque(maxlen=SCORE_WINDOW_SIZE)

    # Enable random exploration to start (based on RANDOM_EXPLORATION_SPAN)
    random_exploration_enabled = True

    # Act and learn over series of episodes
    for episode in range(MAX_EPISODES):
        agents_episode_reward = np.zeros(num_agents)    # A running total of the episode score (per agent)
        done = False                                    # If set to True, episode is finished
        q_loss_log = defaultdict(list)                  # For tracking MSE of DQN/critic (to minimize)
        expected_policy_value_log = defaultdict(list)   # For tracking expected value of policy/actor (to maximize)
        t = 0                                           # For tracking the current timestep within the episode

        # Sample a new noise stdev every episode
        episode_noise_stdev_sample = NOISE_STDEV if random_exploration_enabled else next(noise_stdev_gen)

        # Reset the environment and start a new episode
        state = env_reset(env, brain_name)

        while not done:
            t += 1

            # Convert state np.array to Tensor
            state_tensor = torch.tensor(state).unsqueeze(0).float().to(device)

            # Here, we use random actions for a bit
            # (as mentioned here: https://spinningup.openai.com/en/latest/algorithms/ddpg.html)
            do_random_exploration = episode < RANDOM_EXPLORATION_SPAN
            have_enough_data = len(replay_buffer) >= BATCH_SIZE
            update_on_this_timestep = t % UPDATE_FREQ == 0

            if do_random_exploration:
                actions = process_action(np.random.rand(num_agents, action_size) * 2 - 1, noise_stdev=0.0).tolist()
            else:
                if random_exploration_enabled:
                    print('Disabling random exploration...')
                    random_exploration_enabled = False

                # Figure out what action to take for each agent
                actions = []
                for agent_index, agent in enumerate(agents):
                    # Choose an action by feeding state through "Actor" net
                    agent.policy_net.eval()
                    with torch.no_grad():
                        action = agent.policy_net.forward(state_tensor[:, agent_index, :]) \
                            .cpu().detach().numpy().squeeze(0)
                    action = process_action(action, noise_stdev=episode_noise_stdev_sample)
                    actions.append(action)
                    agent.policy_net.train()

            # Step in env
            next_state, rewards, done = env_step(env, brain_name, actions)
            agents_episode_reward += rewards

            # Store in replay buffer
            replay_buffer.add(state.tolist(), actions, rewards, next_state, done)
            state = next_state

            # Update agents if appropriate
            if update_on_this_timestep and have_enough_data:
                for _ in range(NUM_INNER_UPDATE):
                    for agent_index, agent in enumerate(agents):
                        batch_groups = list(replay_buffer.get_batch(BATCH_SIZE))
                        for k, element in enumerate(batch_groups):
                            batch_groups[k] = element.to(device)

                        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch_groups

                        # Our Policy (Actor) only knows about its local state and own actions
                        agent_reward_batch = reward_batch[:, agent_index, :]
                        # Our Q-Net (Critics) knows about all states and actions
                        env_state_batch = state_batch.reshape(state_batch.size()[0], -1)
                        env_next_state_batch = next_state_batch.reshape(next_state_batch.size()[0], -1)
                        env_action_batch = action_batch.reshape(action_batch.size()[0], -1)

                        # CALCULATE TARGETS
                        # Get Actors' selection for next best action when in next_state
                        _target_next_best_actions = []
                        for inner_agent_index, inner_agent in enumerate(agents):
                            actions = inner_agent.policy_net_target(next_state_batch[:, inner_agent_index, :])
                            if inner_agent_index != agent_index:
                                actions = actions.detach()

                            _target_next_best_actions.append(actions)

                        _target_next_best_actions = torch.cat(_target_next_best_actions, dim=-1)
                        env_target_next_best_actions = torch.tensor(_target_next_best_actions).to(device)

                        # Get Critic's predicted Q value for being in next_state and taking next_best_action)
                        target_q_next_state = agent.q_net_target(env_next_state_batch, env_target_next_best_actions)

                        # Compute the targets and update both Actor and Critic networks
                        # Note: don't want to include the predicted value for next state if we're in a terminal state
                        terminal_state_cancellation = 1 - done_batch
                        target_q = agent_reward_batch + (GAMMA * terminal_state_cancellation * target_q_next_state)

                        # GENERATING PREDICTIONS + APPLYING GRADIENT UPDATES
                        # Update Critic
                        # Minimize mean-squared error between the predicted and target Q(s, a)
                        predicted_q = agent.q_net(env_state_batch, env_action_batch)
                        q_loss = F.mse_loss(predicted_q, target_q)
                        agent.q_optimizer.zero_grad()
                        q_loss.backward()  # Minimize MSE between Q and (bootstrapped) target Q
                        agent.q_optimizer.step()

                        # Update Actor
                        # Calculate Q
                        # Get Actors' selection for next best action when in state
                        _best_actions = []
                        for inner_agent_index, inner_agent in enumerate(agents):
                            actions = inner_agent.policy_net(state_batch[:, inner_agent_index, :])
                            if inner_agent_index != agent_index:
                                actions = actions.detach()
                            _best_actions.append(actions)

                        _best_actions = torch.cat(_best_actions, dim=-1)
                        env_best_actions = torch.tensor(_best_actions, requires_grad=True).to(device)

                        # Disable grads for "Critic" (since we don't need them when updating "Actor")
                        agent.q_net.set_requires_grad(False)

                        # Maximize the expected Q-value of our policy
                        expected_policy_value = torch.mean(agent.q_net(env_state_batch, env_best_actions))
                        agent.policy_optimizer.zero_grad()
                        (-expected_policy_value).backward()
                        agent.policy_optimizer.step()

                        # Re-enable grads for "Critic" for future backprop calculations
                        agent.q_net.set_requires_grad(True)

                        # Log the two losses
                        q_loss_log[agent_index].append(q_loss.item())
                        expected_policy_value_log[agent_index].append(expected_policy_value.item())

                        # UPDATE THE TARGET NETS
                        copy_weights(agent.policy_net, agent.policy_net_target, TAU)
                        copy_weights(agent.q_net, agent.q_net_target, TAU)

        # Episode is done
        episode_score = max(agents_episode_reward)
        scores.append(episode_score)
        scores_window.append(episode_score)
        q_loss_avg = {agent_index: np.mean(loss) if loss else float('inf')
                      for agent_index, loss in q_loss_log.items()} \
                     or {i: float('inf') for i in range(num_agents)}
        q_loss_avg_str = ', '.join(f"Agent {agent}: {loss:.8f}" for agent, loss in q_loss_avg.items())
        expected_policy_value_avg = {agent_index: np.mean(loss) if loss else float('-inf')
                                     for agent_index, loss in expected_policy_value_log.items()} \
                                    or {i: float('-inf') for i in range(num_agents)}
        expected_policy_value_avg_str = ', '.join(
            f"Agent {agent}: {loss:.8f}" for agent, loss in expected_policy_value_avg.items())
        score_mean = np.mean(scores_window) if scores_window else 0.0
        print(f"Episode {episode}/{MAX_EPISODES} ({t} steps): Episode reward: {episode_score:.4f} ; "
              f"Average reward (last 100): {score_mean:.4f} ; "
              f"(Q Loss Avg: [{q_loss_avg_str}] ; E(V) Avg: [{expected_policy_value_avg_str}]) "
              f"[Noise stdev: {episode_noise_stdev_sample}]")

        # Leave the loop if we've reached our goal
        if score_mean >= AVERAGE_SCORE_GOAL:
            print(f"Environment solved at {episode}")
            break

    # Save models
    for agent_index, agent in enumerate(agents):
        torch.save(agent.policy_net.state_dict(), f"policy_net.{agent_index}.pth")
        torch.save(agent.q_net.state_dict(), f"q_net.{agent_index}.pth")

    # Show plot of reward per episode
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.show()


def view_agent(policy_dir: str):
    """Visualizes an episode for pretrained MADDPG agents"""
    hidden_sizes = [512, 256]
    used_batchnorm = True

    # Set up env
    env = UnityEnvironment(file_name=APP_PATH)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    _env_info = env.reset(train_mode=True)[brain_name]  # Note: this is a throwaway reset simply to get info on the env
    num_agents = len(_env_info.agents)
    action_size = brain.vector_action_space_size
    state_size = _env_info.vector_observations.shape[1]

    # Load in policies
    policies = []
    for agent_index in range(num_agents):
        policy_net = PolicyNet(state_size=state_size, action_size=action_size,
                               hidden_sizes=hidden_sizes, use_batchnorm=used_batchnorm)
        policy_net.load_state_dict(torch.load(os.path.join(policy_dir, f"policy_net.{agent_index}.pth")))
        policy_net.eval()
        policies.append(policy_net)

    # Initialize and go through episode
    state = env_reset(env, brain_name, train_mode=False)

    done = False
    agent_rewards = np.zeros(num_agents)
    while not done:
        state_tensor = torch.tensor(state).unsqueeze(0).float()

        actions = []
        for actor_index, policy_net in enumerate(policies):
            action = policy_net.forward(state_tensor[:, actor_index, ]).detach().numpy().squeeze(0)
            actions.append(action)

        next_state, reward, done = env_step(env, brain_name, actions)
        agent_rewards += reward
        state = next_state

    total_reward = max(agent_rewards)
    print(f"Total reward in example episode: {total_reward}")
