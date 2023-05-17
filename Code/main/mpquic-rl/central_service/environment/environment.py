import sys

import numpy as np
import threading
import multiprocessing as mp
import subprocess
import time
import json
import os
import random

import pexpect
from pexpect.popen_spawn import PopenSpawn

from .experiences.quic_web_browse import launchTests
#from utils.logger import config_logger

from central_service.variables import REMOTE_SERVER_RUNNER_HOSTNAME, REMOTE_SERVER_RUNNER_PORT, REMOTE_HOST, REMOTE_PORT

MIDDLEWARE_SOURCE_REMOTE_PATH = "~/go/src/github.com/{username}/middleware"
MIDDLEWARE_BIN_REMOTE_PATH = "~/go/bin/middleware"

class Session:
    '''
        This class loads and parses one by one all configurations
        for our environment!
        It is utilized by both agent and environment
    '''
    def __init__(self, topologies='./environment/topos.json', dgraphs='./environment/train_graphs.json'):
        self._index = 0

        self._topologies, self._len_topo = self.loadTopologies(topologies)
        self._graphs, self._len_graph = self.loadDependencyGraphs(dgraphs)

        self._pairs = self.generatePairs()

    def generatePairs(self):
        tuple_list = []
        for i in range(self._len_topo):
            for j in range(self._len_graph):
                tuple_list.append((i, j))

        return random.sample(tuple_list, len(tuple_list))

    def loadTopologies(self, file):
        topos = []
        with open(file, 'r') as fp:
            #pijus Topos should not be this way
            tmp = """[
                {
                    "paths": [
                        {
                            "bandwidth": "18",
                            "delay": "16.5",
                            "queuingDelay": "0.035"
                        },
                        {
                            "bandwidth": "89",
                            "delay": "19.0",
                            "queuingDelay": "0.096"
                        }
                    ],
                    "netem": [
                        [
                            0,
                            0,
                            "loss 1.27%"
                        ],
                        [
                            1,
                            0,
                            "loss 1.14%"
                        ]
                    ]
                },
                {
                    "paths": [
                        {
                            "bandwidth": "18",
                            "delay": "16.5",
                            "queuingDelay": "0.035"
                        },
                        {
                            "bandwidth": "89",
                            "delay": "19.0",
                            "queuingDelay": "0.096"
                        }
                    ],
                    "netem": [
                        [
                            0,
                            0,
                            "loss 1.27%"
                        ],
                        [
                            1,
                            0,
                            "loss 1.14%"
                        ]
                    ]
                }]"""
            topos = json.load(fp)
            #topos = json.loads(tmp)

        return topos, len(topos)

    def loadDependencyGraphs(self, file):
        graphs = []
        with open(file, 'r') as fp:
            #pijus
            tmp = """[{
                "file": "www.popads.net_",
                "size": 1
            },
            {
                "file": "www.popads.net_",
                "size": 1
            }]"""
            graphs = json.load(fp)
            #graphs = json.loads(tmp)

        output = [elem['file'] for elem in graphs]
        return output, len(output)

    def nextRun(self):
        self._index += 1

        if self._index >= len(self._pairs):
            return -1
        return self._index 

    def getCurrentTopo(self):
        topo = self._topologies[self._pairs[self._index][0]]
        return topo

    def getCurrentGraph(self):
        graph = self._graphs[self._pairs[self._index][1]]
        return graph

    def getCurrentBandwidth(self):
        topo = self.getCurrentTopo()
        return int(topo['paths'][0]['bandwidth']), int(topo['paths'][1]['bandwidth'])


class Environment:
    def __init__(self, bdw_paths, logger, mconfig, remoteHostname=REMOTE_HOST, remotePort=REMOTE_PORT):
        self._totalRuns = 0
        self._logger = logger

        # Session object
        self.session = Session()
        self.curr_topo = Session().getCurrentTopo()
        self.curr_graph = Session().getCurrentGraph()
        self.bdw_paths = bdw_paths

        # Spawn Middleware
        #self._spawn_cmd = self.construct_cmd(mconfig)
        self.mconfig = mconfig
        self._remoteHostname = remoteHostname
        self._remotePort = remotePort
        #self.stop_middleware()
        self.spawn_middleware()

    def spawn_cmd(self):
        return "stdbuf -i0 -o0 -e0 {} -sv {} -cl {} -pub {} -sub {}".format(MIDDLEWARE_BIN_REMOTE_PATH,
                                                    self.mconfig['server'],
                                                    self.mconfig['client'],
                                                    self.mconfig['publisher'],
                                                    self.mconfig['subscriber'])

    def ssh(self, command):
        return f"ssh -p {self._remotePort} {self._remoteHostname} \"{command}\""

    def spawn_middleware(self):
        ''' This method might seem more like a restart.
            First, it kills __if and any__ existing middlewares, and then spawns a new one.
            Small sleep to ensure previous one is killed.
        '''
        self.stop_middleware()
        #time.sleep(0.5)
        #ssh_cmd = ["ssh", "-p", self._remotePort, self._remoteHostname, self._spawn_cmd]
        #ssh_cmd = f"ssh -p {self._remotePort} {self._remoteHostname} {self._spawn_cmd}"
        cmd = self.spawn_cmd() + "2>&1 | tee ~/Workspace/mpquic-sbd/middleware.txt"
        #subprocess.Popen(self.ssh(cmd), bufsize=0,
        #                #stdout=subprocess.PIPE,
        #                #stdout=subprocess.DEVNULL,
        #                #stdout=sys.stdout,
        #                #stderr=subprocess.PIPE,
        #                #stderr=subprocess.DEVNULL,
        #                shell=False)
        child = PopenSpawn(self.ssh(cmd), encoding='utf-8', logfile=sys.stdout)
        print('spawning middleware...')

    def stop_middleware(self, wait=True):
        kill_cmd = f"stdbuf -oL killall {MIDDLEWARE_BIN_REMOTE_PATH}"
        #ssh_cmd = ["ssh", "-p", self._remotePort, self._remoteHostname, kill_cmd]
        sub = subprocess.Popen(self.ssh(kill_cmd),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        #stdout=subprocess.DEVNULL,
                        #stderr=subprocess.DEVNULL,
                        shell=False)
        #if wait:
        #    sub.wait()
        sub.communicate()

    def getNetemToTuple(self, topo):
        '''in json -> tuple (0 0 loss 1.69%) is stored as [0, 0, loss 1.69%]
            revert it back to tuple, otherwise error is produced
        '''
        topo[0]['netem'][0] = (topo[0]['netem'][0][0], topo[0]['netem'][0][1], topo[0]['netem'][0][2])
        topo[0]['netem'][1] = (topo[0]['netem'][1][0], topo[0]['netem'][1][1], topo[0]['netem'][1][2])
        return topo

    def updateEnvironment(self):
        ''' One step update. 
            First load current values, then move to next!
        '''
        topo = [self.session.getCurrentTopo()]
        self.curr_topo = self.getNetemToTuple(topo)
        self.curr_graph = self.session.getCurrentGraph()

        bdw_path1, bdw_path2 = self.session.getCurrentBandwidth()
        self.bdw_paths[0] = bdw_path1
        self.bdw_paths[1] = bdw_path2

        return self.session.nextRun()
        
    def run(self):
        self._totalRuns += 1
        message = "Run Number: {}, Graph: {}" 
        self._logger.info(message.format(self._totalRuns, self.curr_graph))

        print('sbd called')
        self.test_sbd_run()
    def test_sbd_run(self):
        #launchTests(self.curr_topo, self.curr_graph)
        #cmd = ["ssh", "-p", self._remotePort, self._remoteHostname,
        #       "\"sudo python ~/Workspace/mpquic-sbd/network/mininet/build_mininet_router1.py -nm 2 -p 'basic' >> ~/Workspace/mpquic-sbd/test_log.txt\""]
        cmd = self.ssh("sudo python ~/Workspace/mpquic-sbd/network/mininet/build_mininet_router1.py -nm 2 -p 'basic' 2>&1 | tee ~/Workspace/mpquic-sbd/test_log.txt") #
        print(cmd)
        f = open("logs/sbd_log.txt", "a")
        child = PopenSpawn(cmd, timeout=120, logfile=f)
        print('launched sbd test')
        #child.expect("*** Creating network")
        #time.sleep(10)
        child.wait()
        print('sbd run over')
        #the_old_way = subprocess.Popen(cmd,
        #                               stdout=subprocess.PIPE).communicate()[0].rstrip()
        f.close()


    def close(self):
        self.stop_middleware()
        self._logger.info("Environment closing gracefully...")