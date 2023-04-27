#!/usr/bin/env python3

"""
Trains a 2-layer MLP with MetaSGD-VPG.

Usage:

python examples/rl/maml_trpo.py
"""

import random

import cherry as ch
import gymnasium as gym
import numpy as np
import torch
from cherry.algorithms import a2c
from torch import optim
from tqdm import tqdm
import math

from torch import nn
from torch.distributions import Normal, Categorical

import customGym
import learn2learn as l2l


EPSILON = 1e-6

def linear_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()
    return module

class LinearValue(nn.Module):

    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/models/robotics.py)

    **Description**

    A linear state-value function, whose parameters are found by minimizing
    least-squares.

    **Credit**

    Adapted from Tristan Deleu's implementation.

    **References**

    1. Duan et al. 2016. “Benchmarking Deep Reinforcement Learning for Continuous Control.”
    2. [https://github.com/tristandeleu/pytorch-maml-rl](https://github.com/tristandeleu/pytorch-maml-rl)

    **Arguments**

    * **inputs_size** (int) - Size of input.
    * **reg** (float, *optional*, default=1e-5) - Regularization coefficient.

    **Example**
    ~~~python
    states = replay.state()
    rewards = replay.reward()
    dones = replay.done()
    returns = ch.td.discount(gamma, rewards, dones)
    baseline = LinearValue(input_size)
    baseline.fit(states, returns)
    next_values = baseline(replay.next_states())
    ~~~
    """

    def __init__(self, input_size, reg=1e-5):
        super(LinearValue, self).__init__()
        self.linear = nn.Linear(2 * input_size + 4, 1, bias=False)
        self.reg = reg

    def _features(self, states):
        length = states.size(0)
        ones = torch.ones(length, 1).to(states.device)
        al = torch.arange(length, dtype=torch.float32, device=states.device).view(-1, 1) / 100.0
        return torch.cat([states, states**2, al, al**2, al**3, ones], dim=1)

    def fit(self, states, returns):
        features = self._features(states)
        reg = self.reg * torch.eye(features.size(1))
        reg = reg.to(states.device)
        A = features.t() @ features + reg
        b = features.t() @ returns
        coeffs, _, _, _ = torch.linalg.lstsq(b, A)
        #print(self.linear.weight.data.size())
        #print(coeffs.data.t().size())
        self.linear.weight.data = coeffs.data

    def forward(self, states):
        features = self._features(states)
        print(features.size())
        return self.linear(features)


class DiagNormalPolicy(nn.Module):

    def __init__(self, input_size, output_size, hiddens=None, activation='tanh', device='cpu'):
        super(DiagNormalPolicy, self).__init__()
        self.input_size = input_size
        self.device = device
        if hiddens is None:
            hiddens = [100, 100]
        if activation == 'relu':
            activation = nn.ReLU
        elif activation == 'tanh':
            activation = nn.Tanh
        layers = [linear_init(nn.Linear(input_size, hiddens[0])), activation()]
        for i, o in zip(hiddens[:-1], hiddens[1:]):
            layers.append(linear_init(nn.Linear(i, o)))
            layers.append(activation())
        layers.append(linear_init(nn.Linear(hiddens[-1], output_size)))
        layers.append(nn.Softmax())
        self.mean = nn.Sequential(*layers)
        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(1))

    def density(self, state):
        state = state.to(self.device, non_blocking=True)
        loc = self.mean(state)
        #scale = torch.exp(torch.clamp(self.sigma, min=math.log(EPSILON)))
        return Categorical(loc) #Normal(loc=loc, scale=scale)

    def log_prob(self, state, action):
        if not torch.is_tensor(state):
            state = torch.tensor(state).float()
        if not torch.is_tensor(action):
            state = torch.tensor(action).float()
        density = self.density(state)
        return density.log_prob(action).mean()#.mean(dim=1, keepdim=True)

    def forward(self, state):
        if not torch.is_tensor(state):
            print(state)
            state = torch.tensor(state).float()
        density = self.density(state)
        action = torch.clamp(density.sample(), 0, self.input_size-1)
        print(action)
        return action

def compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states):
    # Update baseline
    returns = ch.td.discount(gamma, rewards, dones)
    baseline.fit(states, returns)
    values = baseline(states)
    next_values = baseline(next_states)
    bootstraps = values * (1.0 - dones) + next_values * dones
    next_value = torch.zeros(1, device=values.device)
    return ch.pg.generalized_advantage(tau=tau,
                                       gamma=gamma,
                                       rewards=rewards,
                                       dones=dones,
                                       values=bootstraps,
                                       next_value=next_value)


def maml_a2c_loss(train_episodes, learner, baseline, gamma, tau):
    # Update policy and baseline
    states = train_episodes.state()
    actions = train_episodes.action()
    rewards = train_episodes.reward()
    dones = train_episodes.done()
    next_states = train_episodes.next_state()
    log_probs = learner.log_prob(states, actions)
    advantages = compute_advantages(baseline, tau, gamma, rewards,
                                    dones, states, next_states)
    advantages = ch.normalize(advantages).detach()
    return a2c.policy_loss(log_probs, advantages)


def main(
        experiment='dev',
        env_name='customGym:NetworkEnv',
        adapt_lr=0.1,
        meta_lr=0.01,
        adapt_steps=1,
        num_iterations=5,
        meta_bsz=1,
        adapt_bsz=1,
        tau=1.00,
        gamma=0.99,
        num_workers=1,
        seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    def make_env():
        return gym.make(env_name)

    #env = l2l.gym.AsyncVectorEnv([make_env for _ in range(num_workers)])
    env = make_env()
    #env.seed(seed)
    env = ch.envs.Torch(env)
    #policy = DiagNormalPolicy(env.state_size, env.action_size)
    policy = DiagNormalPolicy(6, 1)#CategoricalPolicy(6, 2)#
    meta_learner = l2l.algorithms.MetaSGD(policy, lr=meta_lr)
    baseline = LinearValue(6)
    opt = optim.Adam(meta_learner.parameters(), lr=meta_lr)
    all_rewards = []

    for iteration in range(num_iterations):
        iteration_loss = 0.0
        iteration_reward = 0.0
        for task_config in tqdm(env.sample_tasks(meta_bsz)):  # Samples a new config
            learner = meta_learner.clone()
            env.set_task(task_config)
            env.reset()
            task = ch.envs.Runner(env)

            # Fast Adapt
            for step in range(adapt_steps):
                train_episodes = task.run(learner, episodes=adapt_bsz)
                loss = maml_a2c_loss(train_episodes, learner, baseline, gamma, tau)
                learner.adapt(loss)

            # Compute Validation Loss
            valid_episodes = task.run(learner, episodes=adapt_bsz)
            loss = maml_a2c_loss(valid_episodes, learner, baseline, gamma, tau)
            iteration_loss += loss
            iteration_reward += valid_episodes.reward().sum().item() / adapt_bsz

        # Print statistics
        print('\nIteration', iteration)
        adaptation_reward = iteration_reward / meta_bsz
        print('adaptation_reward', adaptation_reward)
        all_rewards.append(adaptation_reward)

        adaptation_loss = iteration_loss / meta_bsz
        print('adaptation_loss', adaptation_loss.item())

        opt.zero_grad()
        adaptation_loss.backward()
        opt.step()
    env.close()


if __name__ == '__main__':
    main()