import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import torch
import numpy as np
from torch.autograd import Variable
import pandas as pd

# hyperparameters
hidden_size = 256
learning_rate = 3e-4

# Constants
GAMMA = 0.99
num_steps = 300
max_episodes = 3000


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)
        #self.actor

    def forward(self, state):
        #state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        state = Variable(state.float().unsqueeze(0))
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)

        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist_1 = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist_1

    def reset(self):
        self.apply(ActorCritic.init_weights)

    def calc_a2c_loss(self, Qval, values, rewards, log_probs, entropy_term):

        # Qval = Qval.detach().numpy()[0, 0]
        values = torch.cat(values)
        Qvals = torch.zeros_like(values)  # np.zeros_like(values.detach().numpy())
        # values = values.detach()

        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval

        # update actor critic
        # values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals).detach()
        log_probs = torch.stack(log_probs)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage.detach()).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss #+ entropy_term # learning_rate *
        #loss_history_actor.append(actor_loss.detach().numpy())
        #loss_history_critic.append(critic_loss.detach().numpy())

        return ac_loss

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.1)  ## or simply use your layer.reset_parameters()
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(1 / m.in_features))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(4 / m.in_channels))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

class MetaModel():
    def __init__(self, num_inputs, num_outputs):
        self.actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
        self.ac_optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
    #def predict
    #def a2c(self):
    #    all_lengths = []
    #    average_lengths = []
    #    all_rewards = []
    #    entropy_term = 0
#
    #    for episode in range(max_episodes):
    #        log_probs = []
    #        values = []
    #        rewards = []
#
    #        state = env.reset()
    #        for steps in range(num_steps):
    #            value, policy_dist = self.actor_critic.forward(state)
    #            value = value.detach().numpy()[0, 0]
    #            dist = policy_dist.detach().numpy()
#
    #            action = np.random.choice(num_outputs, p=np.squeeze(dist))
    #            log_prob = torch.log(policy_dist.squeeze(0)[action])
    #            entropy = -np.sum(np.mean(dist) * np.log(dist))
    #            new_state, reward, done, _ = env.step(action)
#
    #            rewards.append(reward)
    #            values.append(value)
    #            log_probs.append(log_prob)
    #            entropy_term += entropy
    #            state = new_state
#
    #            if done or steps == num_steps - 1:
    #                Qval, _ = actor_critic.forward(new_state)
    #                Qval = Qval.detach().numpy()[0, 0]
    #                all_rewards.append(np.sum(rewards))
    #                all_lengths.append(steps)
    #                average_lengths.append(np.mean(all_lengths[-10:]))
    #                if episode % 10 == 0:
    #                    sys.stdout.write(
    #                        "episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode,
    #                                                                                                  np.sum(rewards),
    #                                                                                                  steps,
    #                                                                                                  average_lengths[
    #                                                                                                      -1]))
    #                break
#
    #        # compute Q values
    #        Qvals = np.zeros_like(values)
    #        for t in reversed(range(len(rewards))):
    #            Qval = rewards[t] + GAMMA * Qval
    #            Qvals[t] = Qval
#
    #        # update actor critic
    #        values = torch.FloatTensor(values)
    #        Qvals = torch.FloatTensor(Qvals)
    #        log_probs = torch.stack(log_probs)
#
    #        advantage = Qvals - values
    #        actor_loss = (-log_probs * advantage).mean()
    #        critic_loss = 0.5 * advantage.pow(2).mean()
    #        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term
#
    #        self.ac_optimizer.zero_grad()
    #        ac_loss.backward()
    #        self.ac_optimizer.step()
    #