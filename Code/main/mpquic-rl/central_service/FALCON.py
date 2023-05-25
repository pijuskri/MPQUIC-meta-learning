import dataclasses
import random
from datetime import datetime
import math
from dataclasses import dataclass

import gym
import numpy as np
import scipy
import torch
from avalanche_rl.models.actor_critic import ActorCriticMLP
from matplotlib import pyplot as plt
from torch import optim
import time

from central_service.pytorch_models.metaModel import ActorCritic
from central_service.utils.data_transf import getTrainingVariables
from central_service.utils.logger import config_logger
from customGym.envs.NetworkEnv import NetworkEnv
#from customGym.envs.NetworkEnv import NetworkState
from typing import NamedTuple
from pathlib import Path

from central_service.variables import A_DIM, S_INFO

TRAINING = True #if true, store model after done, have high exploration

# hyperparameters
hidden_size = 256
learning_rate = 0.005#3e-4
apply_loss_steps = 50

GAMMA = 0.99
EPS_START = 0.3 #0.9
EPS_END = 0.05
EPS_DECAY = 1000 #higher = slower decay


class _ChangeFinderAbstract(object):
    def _add_one(self, one, ts, size):
        ts.append(one)
        if len(ts) == size+1:
            ts.pop(0)

    def _smoothing(self, ts):
        return sum(ts)/float(len(ts))


def LevinsonDurbin(r, lpcOrder):
    """
    from http://aidiary.hatenablog.com/entry/20120415/1334458954
    """
    a = np.zeros(lpcOrder + 1, dtype=np.float64)
    e = np.zeros(lpcOrder + 1, dtype=np.float64)

    a[0] = 1.0
    a[1] = - r[1] / r[0]
    e[1] = r[0] + r[1] * a[1]
    lam = - r[1] / r[0]

    for k in range(1, lpcOrder):
        lam = 0.0
        for j in range(k + 1):
            lam -= a[j] * r[k + 1 - j]
        lam /= e[k]

        U = [1]
        U.extend([a[i] for i in range(1, k + 1)])
        U.append(0)

        V = [0]
        V.extend([a[i] for i in range(k, 0, -1)])
        V.append(1)

        a = np.array(U) + lam * np.array(V)
        e[k + 1] = e[k] * (1.0 - lam * lam)

    return a, e[-1]

class _SDAR_1Dim(object):
    def __init__(self, r, order):
        self._r = r
        self._mu = np.random.random()
        self._sigma = np.random.random()
        self._order = order
        self._c = np.random.random(self._order+1) / 100.0

    def update(self, x, term):
        assert len(term) >= self._order, "term must be order or more"
        term = np.array(term)
        self._mu = (1.0 - self._r) * self._mu + self._r * x
        for i in range(1, self._order + 1):
            self._c[i] = (1 - self._r) * self._c[i] + self._r * (x - self._mu) * (term[-i] - self._mu)
        self._c[0] = (1-self._r)*self._c[0]+self._r * (x-self._mu)*(x-self._mu)
        what, e = LevinsonDurbin(self._c, self._order)
        xhat = np.dot(-what[1:], (term[::-1] - self._mu))+self._mu
        self._sigma = (1-self._r)*self._sigma + self._r * (x-xhat) * (x-xhat)
        return -math.log(math.exp(-0.5*(x-xhat)**2/self._sigma)/((2 * math.pi)**0.5*self._sigma**0.5)), xhat

class ChangeFinder(_ChangeFinderAbstract):
    def __init__(self, r=0.5, order=1, smooth=7):
        assert order > 0, "order must be 1 or more."
        assert smooth > 2, "term must be 3 or more."
        self._smooth = smooth
        self._smooth2 = int(round(self._smooth/2.0))
        self._order = order
        self._r = r
        self._ts = []
        self._first_scores = []
        self._smoothed_scores = []
        self._second_scores = []
        self._sdar_first = _SDAR_1Dim(r, self._order)
        self._sdar_second = _SDAR_1Dim(r, self._order)

    def update(self, x):
        score = 0
        predict = x
        predict2 = 0
        if len(self._ts) == self._order:  # 第一段学習
            score, predict = self._sdar_first.update(x, self._ts)
            self._add_one(score, self._first_scores, self._smooth)
        self._add_one(x, self._ts, self._order)
        second_target = None
        if len(self._first_scores) == self._smooth:  # 平滑化
            second_target = self._smoothing(self._first_scores)
        if second_target and len(self._smoothed_scores) == self._order:  # 第二段学習
            score, predict2 = self._sdar_second.update(second_target, self._smoothed_scores)
            self._add_one(score,
                          self._second_scores, self._smooth2)
        if second_target:
            self._add_one(second_target, self._smoothed_scores, self._order)
        if len(self._second_scores) == self._smooth2:
            return self._smoothing(self._second_scores), predict
        else:
            return 0.0, predict


@dataclass
class NetworkState:
    normalized_bwd_path0: float
    normalized_bwd_path1: float
    normalized_srtt_path0: float
    normalized_srtt_path1: float
    normalized_loss_path0: float
    normalized_loss_path1: float

@dataclass
class SavedModel:
    model: dict
    start: NetworkState
    end: NetworkState

class FalconMemory:
    def __init__(self):
        self.lookback = 3
        self.observations = []
        self.cf = []
        self.models: list[SavedModel] = []
        for i in range(S_INFO):
            #self.cf.append(ChangeFinder(r=0.5, order=1, smooth=3))#))
            self.cf.append(ChangeFinder(r=0.4, order=1, smooth=3))  # ))
    def findModel(self, cur_state):
        for model in self.models:
            #print(model)
            start = dataclasses.astuple(model.start)
            end = dataclasses.astuple(model.end)
            ranges = [(min(start[i], end[i]), max(start[i], end[i])) for i in range(len(start))]
            within_range = True
            for i in range(len(ranges)):
                min_v, max_v = ranges[i]
                within_range = within_range and min_v <= cur_state[i] <= max_v
            if within_range:
                return model.model
        return None # found no matching model
    def add_model(self, start, end, model_state):
        self.models.append(SavedModel(model_state, NetworkState(*start), NetworkState(*end)))
    def add_obs(self, obs, model_state):
        change = False
        probs = np.zeros(S_INFO)
        for i, value in enumerate(obs):
            probs[i], _ = self.cf[i].update(value)
        #prob = np.cumprod(probs)

        #print(obs)
        print('network switch prob {:.1f} {:.1f}'.format(np.mean(probs), np.cumprod(probs)[-1])) #, prob
        prob = np.cumprod(probs)[-1]
        #print()
        #TODO after change have cooldown to avoid constant switching
        if prob > 99999.5: #0.5
            start = self.observations[0]
            end = self.observations[-3]
            change = True
            self.observations = [self.observations[-2], self.observations[-1]]
            self.add_model(start, end, model_state)
        self.observations.append(obs)
        return change

def test_change_detect():
    before = [0.74747475, 0.04040404, 0.50918333, 0.16425, 0.1, 0.15]
    after = [0.84848485, 0.57575758, 0.0, 0.0, 0.0, 0.0]
    memory = FalconMemory()
    for i in range(20):
        memory.add_obs(before, None)
    print("=================New env================")
    for i in range(10):
        memory.add_obs(after, None)
class ReplayMemory:
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.actions = []
        self.entropy_term = []
        #self.logger = logger
    def update_memory(self, action, value, policy_dist, reward):
        dist = policy_dist.detach().numpy()
        log_prob = torch.log(policy_dist.squeeze(0)[action])
        entropy = -np.sum(np.mean(dist) * np.log(dist))

        self.actions.append(action)

        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.entropy_term.append(entropy)
        #self.entropy_term += entropy
    def get_memory(self, length=-1):
        if length < 0:
            length = len(self.log_probs)
        entropy_term = np.sum(self.entropy_term[-length:])
        #Qval, values, rewards, log_probs, entropy_term
        return self.values[-1], self.values[-length:], self.rewards[-length:], self.log_probs[-length:], entropy_term

    def clear(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.actions = []
        self.entropy_term = []

    def __len__(self):
        return len(self.values)


loss_history_actor = []
loss_history_critic = []
model_name = 'FALCON'
#def get_epsilon(self, t):
#    return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))
def calc_a2c_loss(Qval, values, rewards, log_probs, entropy_term):

    #Qval = Qval.detach().numpy()[0, 0]
    values = torch.cat(values)
    Qvals = torch.zeros_like(values) #np.zeros_like(values.detach().numpy())
    #values = values.detach()

    for t in reversed(range(len(rewards))):
        Qval = rewards[t] + GAMMA * Qval
        Qvals[t] = Qval

    # update actor critic
    #values = torch.FloatTensor(values)
    Qvals = torch.FloatTensor(Qvals).detach()
    log_probs = torch.stack(log_probs)

    advantage = Qvals - values
    actor_loss = (-log_probs * advantage.detach()).mean()
    critic_loss = 0.5 * advantage.pow(2).mean()
    ac_loss = actor_loss + critic_loss #+ entropy_term # learning_rate *
    loss_history_actor.append(actor_loss.detach().numpy())
    loss_history_critic.append(critic_loss.detach().numpy())

    return ac_loss
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    num_inputs = S_INFO
    num_outputs = A_DIM
    max_steps = 100000000

    run_id = datetime.now().strftime('%Y%m%d_%H_%M_%S')+f"_{model_name}"
    log_dir = Path("runs/"+run_id)
    log_dir.mkdir(parents=True)

    #actor_critic = ActorCriticMLP(num_inputs, num_outputs, hidden_size, hidden_size)
    print('other model', ActorCriticMLP(num_inputs, num_outputs, hidden_size, hidden_size))
    actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)
    print('model', actor_critic)

    all_lengths = []
    average_lengths = []
    all_rewards = []
    states = []
    loss_history = []

    #entropy_term = 0
    total_steps = 0

    env = gym.make('NetworkEnv')
    memory = FalconMemory()
    replay_memory = ReplayMemory()

    logger = config_logger('agent', './logs/agent.log')
    logger.info("Run Agent until training stops...")

    start_time = time.time()
    print("Starting agent")
    with torch.autograd.set_detect_anomaly(True):
        for episode in range(1):
            state = env.reset()
            start_time = time.time()
            print("Episode ", episode)
            #TODO handle max steps
            for step in range(max_steps):

                value, policy_dist = actor_critic.forward(torch.Tensor(state))
                #value = value.detach().numpy()[0, 0]#[0, 0]
                #print(value)
                dist = policy_dist.detach().numpy()

                sample = random.random()
                eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                                math.exp(-1. * total_steps / EPS_DECAY)
                if sample > eps_threshold:
                    action = np.random.choice(num_outputs, p=np.squeeze(dist))
                else:
                    action = env.action_space.sample()

                new_state, reward, done, info = env.step(action)
                state = new_state

                replay_memory.update_memory(action, value, policy_dist, reward)

                change = memory.add_obs(state, actor_critic.state_dict())
                #if change:
                #    found = memory.findModel(state)
                #    if found is not None:
                #        actor_critic.load_state_dict(found)
                #    else: actor_critic.reset()
                #    replay_memory.clear()
                states.append(state)

                #TODO make sure replay memory loss is used for previous model after change reset
                if total_steps % apply_loss_steps == 0 and total_steps > 0 and len(replay_memory) > apply_loss_steps:
                    memory_values = replay_memory.get_memory(apply_loss_steps)
                    ac_loss = calc_a2c_loss(*memory_values)
                    ac_optimizer.zero_grad()
                    ac_loss.backward()
                    ac_optimizer.step()
                    loss_history.append(ac_loss.detach().numpy())
                    msg = "TD_loss: {}, Avg_reward: {}, Avg_entropy: {}".format(ac_loss, np.mean(memory_values[2]),
                                                                                np.mean(replay_memory.entropy_term))
                    logger.debug(msg)

                total_steps += 1

                if done:
                    break

            #if len(log_probs) == 0:
            #    continue

            # compute Q values
            #Qval, _ = actor_critic.forward(torch.tensor(state))
            logger.debug("====")
            logger.debug("Epoch: {}".format(episode))
            #msg = "TD_loss: {}, Avg_reward: {}, Avg_entropy: {}".format(ac_loss, np.mean(replay_memory.rewards),
            #                                                            entropy_term)
            #logger.debug(msg)
            logger.debug("====")
    end_time = time.time()
    np.savetxt(log_dir / "rewards.csv",np.array(replay_memory.rewards),delimiter=", ",fmt='% s')
    np.savetxt(log_dir / "loss.csv", np.array(loss_history), delimiter=", ", fmt='% s')
    np.savetxt(log_dir / "states.csv", np.array(states), delimiter=", ", fmt='% s')
    plt.plot(moving_average(replay_memory.rewards, 30))
    plt.title(f'Reward {run_id}')
    plt.show()
    plt.plot(moving_average(loss_history, 3))
    #plt.yscale('log')
    plt.yscale('log')
    plt.title(f'Loss {run_id}')
    #plt.plot(moving_average(loss_history_critic, 1))
    #plt.plot(moving_average(loss_history_actor, 1))
    plt.show()
    env.close()

    print("steps/",total_steps/(end_time-start_time))

#def moving_average(a, n=3):
#    ret = np.cumsum(a, dtype=float)
#    ret[n:] = ret[n:] - ret[:-n]
#    return ret[n - 1:] / n

#def moving_average(x, w):
#    return np.convolve(x, np.ones(w), 'valid') / w
#def moving_average(x, w):
#    return scipy.ndimage.gaussian_filter1d(x, (np.std(x) * w * 5)/(len(x)))
def plot_most_recent_results():
    pass
def moving_average(x, w):
    if w == 1:
        return x
    m = np.pad(x, int(w/2), mode='mean', stat_length=int(w/2)) #constant_values=np.mean(x)
    return scipy.ndimage.gaussian_filter1d(m, np.std(x) * w * 2) #(np.std(x) * w * 25)/(len(x)
if __name__ == '__main__':
    main()
    #data = np.genfromtxt('logs/rewards.csv', delimiter=',')
    #print(len(data))
    #plt.plot(moving_average(data, 20))
    #plt.show()
    #test_change_detect()


