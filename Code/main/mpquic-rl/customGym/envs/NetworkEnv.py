import numpy as np

import gymnasium as gym
from gymnasium import spaces

# global imports
import dataclasses
from dataclasses import dataclass
import os
import threading, queue
import multiprocessing as mp

from central_service.pytorch_models.metaModel import MetaModel, ActorCritic
from central_service.variables import REMOTE_HOST

import time

# local imports
from central_service.centraltrainer.request_handler import RequestHandler
from central_service.centraltrainer.collector import Collector
from central_service.environment.environment import Environment
from central_service.utils.logger import config_logger
from central_service.utils.queue_ops import get_request, put_response
from central_service.utils.data_transf import arrangeStateStreamsInfo, getTrainingVariables, allUnique

# ---------- Global Variables ----------
S_INFO = 6  # bandwidth_path_i, path_i_mean_RTT, path_i_retransmitted_packets + path_i_lost_packets
S_LEN = 8  # take how many frames in the past
A_DIM = 2 # two actions -> path 1 or path 2
PATHS = [1, 3] # correspond to path ids
DEFAULT_PATH = 1  # default path without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000

SUMMARY_DIR = '../../central_service/logs/'
LOG_FILE = '../../central_service/logs/tf_log.log'

SSH_HOST = 'localhost'


def environment(bdw_paths: mp.Array, stop_env: mp.Event, end_of_run: mp.Event):
    rhostname = REMOTE_HOST  # 'mininet' + '@' + SSH_HOST

    config = {
        'server': 'ipc:///tmp/zmq',
        'client': 'tcp://*:5555',
        'publisher': 'tcp://*:5556',
        'subscriber': 'ipc:///tmp/pubsub'
    }
    logger = config_logger('environment', filepath='../../central_service/logs/environment.log')
    env = Environment(bdw_paths, logger=logger, mconfig=config, remoteHostname=rhostname)

    # Lets measure env runs in time
    while not stop_env.is_set():

        # Only the agent can unblock this loop, after a training-batch has been completed
        while not end_of_run.is_set():
            try:
                # update environment config from session
                if env.updateEnvironment() == -1:
                    stop_env.set()
                    end_of_run.set()
                    break

                # run a single session & measure
                # -------------------
                now = time.time()
                env.run()
                end = time.time()
                # -------------------

                diff = int(end - now)
                logger.debug("Time to execute one run: {}s".format(diff))

                end_of_run.set()  # set the end of run so our agent knows
                # env.spawn_middleware() # restart middleware
            except Exception as ex:
                logger.error(ex)
                break
        time.sleep(0.1)

    env.close()


@dataclass
class NetworkState:
    normalized_bwd_path0: float
    normalized_bwd_path1: float
    normalized_srtt_path0: float
    normalized_srtt_path1: float
    normalized_loss_path0: float
    normalized_loss_path1: float

class NetworkEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self):

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Tuple((
            spaces.Box(0, 100, shape=(1,), dtype=np.float32), #band
            spaces.Box(0, 100, shape=(1,), dtype=np.float32), #band
            spaces.Box(0, 100, shape=(1,), dtype=np.float32), #RTT
            spaces.Box(0, 100, shape=(1,), dtype=np.float32), #RTT
            spaces.Box(0, 30, shape=(1,), dtype=np.float32), #Loss
            spaces.Box(0, 30, shape=(1,), dtype=np.float32), #Loss
        ))

        self.action_space = spaces.Discrete(A_DIM)
        spaces.Space()

        np.random.seed(RANDOM_SEED)

        # Create results path
        if not os.path.exists(SUMMARY_DIR):
            os.makedirs(SUMMARY_DIR)

        # Spawn request handler
        self.tqueue = queue.Queue(1)
        self.rhandler = RequestHandler(1, "rhandler-thread", tqueue=self.tqueue, host=SSH_HOST, port='5555')
        self.rhandler.start()

        # Spawn collector thread
        self.cqueue = queue.Queue(0)
        self.collector = Collector(2, "collector-thread", queue=self.cqueue, host=SSH_HOST, port='5556')
        self.collector.start()

        # Spawn environment # process -- not a thread
        self.bdw_paths = mp.Array('i', 2)
        self.stop_env = mp.Event()
        self.end_of_run = mp.Event()
        env = mp.Process(target=environment, args=(self.bdw_paths, self.stop_env, self.end_of_run))
        env.start()

        # keep record of threads and processes
        self.tp_list = [self.rhandler, self.collector, env]

        self.logger = config_logger('agent', './logs/agent.log')
        self.logger.info("Run Agent until training stops...")

        #THESE SHOULD BE RESET EACH EPISODE
        self.request = None

    def get_net_state(self) -> NetworkState:
        path1_smoothed_RTT, path1_bandwidth, path1_packets, \
            path1_retransmissions, path1_losses, \
            path2_smoothed_RTT, path2_bandwidth, path2_packets, \
            path2_retransmissions, path2_losses, \
            = getTrainingVariables(self.request)

        normalized_bwd_path0 = (self.bdw_paths[0] - 1.0) / (100.0 - 1.0)
        normalized_bwd_path1 = (self.bdw_paths[1] - 1.0) / (100.0 - 1.0)
        normalized_srtt_path0 = ((path1_smoothed_RTT * 1000.0) - 1.0) / (120.0)
        normalized_srtt_path1 = ((path2_smoothed_RTT * 1000.0) - 1.0) / (120.0)
        normalized_loss_path0 = ((path1_retransmissions + path1_losses) - 0.0) / 20.0
        normalized_loss_path1 = ((path2_retransmissions + path2_losses) - 0.0) / 20.0
        return NetworkState(normalized_bwd_path0,normalized_bwd_path1,normalized_srtt_path0,normalized_srtt_path1,
                            normalized_loss_path0,normalized_loss_path1)
    def env_send(self, request, path):
        action_vec = np.zeros(A_DIM)
        action_vec[path] = 1
        self.logger.debug("PATH: {}".format(path))

        # prepare response
        response = [request['StreamID'], PATHS[path]]
        response = [str(r).encode('utf-8') for r in response]
        ev2 = threading.Event()
        put_response((response, ev2), self.tqueue, self.logger)
        ev2.wait()  # blocks until `consumer` (i.e. rh) receives response

# %%
# Constructing Observations From Environment States
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Since we will need to compute observations both in ``reset`` and
# ``step``, it is often convenient to have a (private) method ``_get_obs``
# that translates the environment’s state into an observation. However,
# this is not mandatory and you may as well compute observations in
# ``reset`` and ``step`` separately:

    def _get_obs(self):
        if self.request is not None:
            ret_state = self.get_net_state()
            return list(dataclasses.astuple(ret_state))
        else:
            raise Exception('Can not get observation when request is null')

# %%
# We can also implement a similar method for the auxiliary information
# that is returned by ``step`` and ``reset``. In our case, we would like
# to provide the manhattan distance between the agent and the target:

    def _get_info(self):
        #return {
        #    "distance": np.linalg.norm(
        #        self._agent_location - self._target_location, ord=1
        #    )
        #}
        return None

    def reward(self, action, completed):
        action_vec = np.zeros(A_DIM)
        action_vec[action] = 1

        s = self.get_net_state()
        aggr_srtt = s.normalized_srtt_path0 + s.normalized_srtt_path1
        aggr_loss = s.normalized_loss_path0 + s.normalized_loss_path1

        completed_factor = 0
        if completed:
            stream_info = []
            with self.cqueue.mutex:
                for elem in list(self.cqueue.queue):
                    #stream_info.append(elem)
                    completed_factor += elem['CompletionTime']
            #stream_info['CompletionTime']

        reward = (action_vec[0] * s.normalized_bwd_path0 + action_vec[1] * s.normalized_bwd_path1) - completed_factor - (0.8 * aggr_srtt) - (1.0 * aggr_loss)
        return reward

# %%
# Oftentimes, info will also contain some data that is only available
# inside the ``step`` method (e.g. individual reward terms). In that case,
# we would have to update the dictionary that is returned by ``_get_info``
# in ``step``.

# %%
# Reset
# ~~~~~
#
# The ``reset`` method will be called to initiate a new episode. You may
# assume that the ``step`` method will not be called before ``reset`` has
# been called. Moreover, ``reset`` should be called whenever a done signal
# has been issued. Users may pass the ``seed`` keyword to ``reset`` to
# initialize any random number generator that is used by the environment
# to a deterministic state. It is recommended to use the random number
# generator ``self.np_random`` that is provided by the environment’s base
# class, ``gymnasium.Env``. If you only use this RNG, you do not need to
# worry much about seeding, *but you need to remember to call
# ``super().reset(seed=seed)``* to make sure that ``gymnasium.Env``
# correctly seeds the RNG. Once this is done, we can randomly set the
# state of our environment. In our case, we randomly choose the agent’s
# location and the random sample target positions, until it does not
# coincide with the agent’s position.
#
# The ``reset`` method should return a tuple of the initial observation
# and some auxiliary information. We can use the methods ``_get_obs`` and
# ``_get_info`` that we implemented earlier for that:

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)


        info = self._get_info()

        self.end_of_run.clear()

        with self.cqueue.mutex:
            # clear the queue
            self.cqueue.queue.clear()

        if self.request is None:
            self.request, ev1 = get_request(self.tqueue, self.logger, end_of_run=self.end_of_run)
            ev1.set()  # let `producer` (rh) know we received request

        observation = self._get_obs()

        return observation, info

# %%
# Step
# ~~~~
#
# The ``step`` method usually contains most of the logic of your
# environment. It accepts an ``action``, computes the state of the
# environment after applying that action and returns the 4-tuple
# ``(observation, reward, done, info)``. Once the new state of the
# environment has been computed, we can check whether it is a terminal
# state and we set ``done`` accordingly. Since we are using sparse binary
# rewards in ``GridWorldEnv``, computing ``reward`` is trivial once we
# know ``done``. To gather ``observation`` and ``info``, we can again make
# use of ``_get_obs`` and ``_get_info``:

    def step(self, action):

        self.env_send(self.request, action)

        #terminated =
        reward = self.reward(action, False)
        observation = self._get_obs()
        info = self._get_info()

        request, ev1 = get_request(self.tqueue, self.logger, end_of_run=self.end_of_run)

        if request is None and self.end_of_run.is_set():
            reward = self.reward(action, True)
            return observation, reward, True, False, info

        self.request = request
        ev1.set()  # let `producer` (rh) know we received request

        return observation, reward, False, False, info

    #def _done(self):


# %%
# Rendering
# ~~~~~~~~~
#
# Here, we are using PyGame for rendering. A similar approach to rendering
# is used in many environments that are included with Gymnasium and you
# can use it as a skeleton for your own environments:

    def render(self):
        return


# %%
# Close
# ~~~~~
#
# The ``close`` method should close any open resources that were used by
# the environment. In many cases, you don’t actually have to bother to
# implement this method. However, in our example ``render_mode`` may be
# ``"human"`` and we might need to close the window that has been opened:

    def close(self):
        self.stop_env.set()
        self.rhandler.stophandler()
        self.collector.stophandler()

        # wait for threads and process to finish gracefully...
        for tp in self.tp_list:
            tp.join()