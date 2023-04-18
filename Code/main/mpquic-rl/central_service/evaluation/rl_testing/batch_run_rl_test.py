# global imports
import json
import random
import time
import os

from central_service.variables import REMOTE_SERVER_RUNNER_PORT
# local imports
from experiences.quic_web_browse import launchTests


# global vars
TOPOS_FP = './../topos.json'
GRAPHS_FP = './../test_graphs.json'
TG_PAIRS = ['']


random.seed(42)


def getNetemToTuple(topo):
    '''in json -> tuple (0 0 loss 1.69%) is stored as [0, 0, loss 1.69%]
        revert it back to tuple, otherwise error is produced
    '''
    topo[0]['netem'][0] = (topo[0]['netem'][0][0], topo[0]['netem'][0][1], topo[0]['netem'][0][2])
    topo[0]['netem'][1] = (topo[0]['netem'][1][0], topo[0]['netem'][1][1], topo[0]['netem'][1][2])
    return topo


def load_or_generate_pairs(file):
    if not os.path.isfile(file):
        with open(TOPOS_FP, 'r') as fp:
            topos = json.load(fp)
        with open(GRAPHS_FP, 'r') as fp:
            graphs = json.load(fp)

        tuple_list = []
        for i in range(len(topos)):
            for j in range(len(graphs)):
                tuple_list.append((i, j))

        pairs = random.sample(tuple_list, 20)
        output = []
        for (t, g) in pairs:
            pair = {
                'graph': graphs[g],
                'topo': topos[t]
            }
            output.append(pair)
        with open(file, 'w') as fp:
            json.dump(output, fp, indent=4)
        return output
    else:
        with open(file, 'r') as fp:
            pairs = json.load(fp)
        return pairs       
            

def restart_nn_inference(bdw_path1, bdw_path2):
    PYTHON_NN_INFERENCE = 'python3.6'

    def send_cmd(cmd):
        import subprocess
        ssh_cmd = ["ssh", "-p", REMOTE_PORT, REMOTE_HOST, cmd]
        subprocess.Popen(ssh_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        shell=False)

    send_cmd("killall {}".format(PYTHON_NN_INFERENCE))
    time.sleep(0.5)
    spawn_cmd = "{} {} {} {}".format("python3.6", "./git/nn_testing/nn_inference.py",str(bdw_path1), str(bdw_path2))
    send_cmd(spawn_cmd)


def main():

    pairs = load_or_generate_pairs(TG_PAIRS[0])
    counter = 1
    with open('', 'w') as fp:
        for p in pairs:
            graph = p['graph']['file']
            topo = getNetemToTuple([p['topo']])

            bdw_path1 = p['topo']['paths'][0]['bandwidth']
            bdw_path2 = p['topo']['paths'][1]['bandwidth'] 

            fp.write("{},\t{},\t{}\n".format(counter, graph, p['topo']))
            counter += 1
            restart_nn_inference(bdw_path1, bdw_path2)

            for _ in range(10):
                try:
                    start = time.time()
                    launchTests(topo, graph)
                    end = time.time()

                    diff = int(start - end)
                    print("runtime: {}s".format(diff))
                except Exception as ex:
                    print (ex)


    pairs = load_or_generate_pairs(TG_PAIRS[1])
    counter = 1
    with open('', 'w') as fp:
        for p in pairs:
            graph = p['graph']['file']
            topo = getNetemToTuple([p['topo']])

            bdw_path1 = p['topo']['paths'][0]['bandwidth']
            bdw_path2 = p['topo']['paths'][1]['bandwidth']

            fp.write("{},\t{},\t{}\n".format(counter, graph, p['topo']))
            counter += 1
            restart_nn_inference(bdw_path1, bdw_path2)

            for _ in range(10):
                try:
                    start = time.time()
                    launchTests(topo, graph)
                    end = time.time()

                    diff = int(start - end)
                    print("runtime: {}s".format(diff))
                except Exception as ex:
                    print (ex)

    pairs = load_or_generate_pairs(TG_PAIRS[2])
    counter = 1
    with open('', 'w') as fp:
        for p in pairs:
            graph = p['graph']['file']
            topo = getNetemToTuple([p['topo']])

            bdw_path1 = p['topo']['paths'][0]['bandwidth']
            bdw_path2 = p['topo']['paths'][1]['bandwidth']

            fp.write("{},\t{},\t{}\n".format(counter, graph, p['topo']))
            counter += 1
            restart_nn_inference(bdw_path1, bdw_path2)

            for _ in range(10):
                try:
                    start = time.time()
                    launchTests(topo, graph)
                    end = time.time()

                    diff = int(start - end)
                    print("runtime: {}s".format(diff))
                except Exception as ex:
                    print (ex)


if __name__ == "__main__":
    main()