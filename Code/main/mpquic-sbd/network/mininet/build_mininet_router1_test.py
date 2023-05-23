#!/usr/bin/python
#this is python2 code!!!!!!
import time
import sys

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Node
from mininet.log import setLogLevel, info
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.util import dumpNodeConnections
from datetime import datetime
#import subprocess

import argparse
from mininet.link import TCLink

#http://recolog.blogspot.com/2016/02/emulating-networks-with-routers-using.html
 
with_background = 1
number_of_interface_client = 2
download = False
playback = 'basic'
PATH_DIR = "/Workspace/mpquic-sbd/"

USER='mininet'


TC_QDISC_RATE = 1.5 #Mbit
TC_QDISC_LATENCY = 20 #ms
TC_QDISC_BURST = 2560
NICE = 'nice -n -10'
CLIENT = 'CLIENT'
SERVER = 'SERVER'
TIMEOUT = 35
TCP_CORE_MB = 100000

class LinuxRouter( Node ):
    "A Node with IP forwarding enabled."

    def config( self, **params ):
        super( LinuxRouter, self).config( **params )
        # Enable forwarding on the router
        self.cmd( 'sysctl net.ipv4.ip_forward=1' )

    def terminate( self ):
        self.cmd( 'sysctl net.ipv4.ip_forward=0' )
        super( LinuxRouter, self ).terminate()


class NetworkTopo( Topo ):
    "A LinuxRouter connecting three IP subnets"

    def build( self, **_opts ):

        #r1 = self.addHost('r1', cls=LinuxRouter, ip='10.0.0.1/30')
        #r2 = self.addHost('r2', cls=LinuxRouter, ip='10.0.0.2/30')
        #r3 = self.addHost('r3', cls=LinuxRouter, ip='10.0.0.5/30')
        #r4 = self.addHost('r4', cls=LinuxRouter, ip='10.0.0.6/30')
        #r5 = self.addHost('r5', cls=LinuxRouter, ip='10.0.0.9/30')
        #client = self.addHost('client', ip='10.0.1.2/24', defaultRoute='via 10.0.1.1')
        #server = self.addHost('server', ip='10.0.2.2/24', defaultRoute='via 10.0.2.1')
        #client1 = self.addHost('client1', ip='10.0.5.2/24', defaultRoute='via 10.0.5.1')
        #client2 = self.addHost('client2', ip='10.0.7.2/24', defaultRoute='via 10.0.7.1')
        #server1 = self.addHost('server1', ip='10.0.6.2/24', defaultRoute='via 10.0.6.1')
        #server2 = self.addHost('server2', ip='10.0.8.2/24', defaultRoute='via 10.0.8.1')
        
        # linkopts = dict(bw=10, delay='5ms', loss=10, max_queue_size=1000, use_htb=True)
        #self.addLink(r1, r2, intfName1='r1-eth0', intfName2='r2-eth0')
        #self.addLink(r3, r4, intfName1='r3-eth0', intfName2='r4-eth0')
        #self.addLink(r2, r5, intfName1='r2-eth1', params1={ 'ip' : '10.0.0.9/30' }, intfName2='r5-eth0', params2={ 'ip' : '10.0.0.10/30' })
        #self.addLink(r4, r5, intfName1='r4-eth1', params1={ 'ip' : '10.0.0.13/30' }, intfName2='r5-eth1', params2={ 'ip' : '10.0.0.14/30' })
        #
        ## client
        #self.addLink( client, r1, intfName2='r1-eth1', params2={ 'ip' : '10.0.1.1/24' } )
        #self.addLink( client, r3, intfName1='client-eth1', params1={ 'ip' : '10.0.3.2/24' }, intfName2='r3-eth1', params2={ 'ip' : '10.0.3.1/24' } )
#
        ## server
        #self.addLink( server, r5, intfName2='r5-eth2', params2={ 'ip' : '10.0.2.1/24' } )

        client = self.addHost('client', ip='10.0.1.2/24')
        server = self.addHost('server', ip='10.0.2.2/24')

        s1 = self.addSwitch('s1')
        self.addLink(client, s1, bw=100, delay="10ms")
        self.addLink(server, s1, bw=100, delay="10ms")

        s2 = self.addSwitch('s2')
        self.addLink(client, s2, bw=100, delay="20ms")
        self.addLink(server, s2, bw=100, delay="20ms")

def run():
    "Test linux router"
    topo = NetworkTopo()

    #subprocess.run("kill $(lsof - t - i:6633)")
    net = Mininet( topo=topo, waitConnected=True, link=TCLink)  # controller is used by s1-s3
    net.start()

    # default route for the selection process of normal internet-traffic
    #net[ 'client' ].cmd("ip route add default scope global nexthop via 10.0.1.1 dev client-eth0")

    #configuration server
    # This creates two different routing tables, that we use based on the source-address.
    #net[ 'server' ].cmd("ip rule add from 10.0.2.2 table 1")
    #
    ## Configure the two different routing tables
    #net[ 'server' ].cmd("ip route add 10.0.2.0/24 dev server-eth0 scope link table 1")
    #net[ 'server' ].cmd("ip route add default via 10.0.2.1 dev server-eth0 table 1")
#
    ## default route for the selection process of normal internet-traffic
    #net[ 'server' ].cmd("ip route add default scope global nexthop via 10.0.2.1 dev server-eth0")

    print( "Dumping host connections")
    dumpNodeConnections( net.hosts )
    print( "Testing network connectivity")
    net.pingAll()
    print( "Testing bandwidth between client and h4")

    info( '*** Routing Table on Router:\n' )
    h1, h4 = net.getNodeByName('client', 'server')
    #net.iperf((h1, h4), seconds=5)
    # print net[ 'r1' ].cmd( 'route' )    
    
    #Run experiment
    #print( net[ 'server' ].cmd("cd /home/" + USER + PATH_DIR))
    #print( net[ 'server' ].cmd("pwd"))
    #
#
    #net[ 'client' ].cmd("cd /home/" + USER + PATH_DIR)
    ## print net[ 'client' ].cmd("./remove_files.sh")
#
    #net[ 'server' ].cmd("nice -n -10 src/dash/caddy/caddy -conf /home/" + USER + "/Caddyfile -quic -mp &>> out &")
    #server_pid = int(net['server'].cmd('echo $!'))
    #
    #time.sleep(5)
    #file_mpd = 'output_dash.mpd'
    ##if playback == 'sara':
    ##    file_mpd = 'output_dash2.mpd'
    #
    #start = datetime.now()
    #file_out = 'data/out_{0}_{1}.txt'.format(playback, start.strftime("%Y-%m-%d.%H:%M:%S"))
#
    #for i in range(n_times):
    #    if download:
    #        cmd = "nice -n -10 python3 src/AStream/dist/client/dash_client.py -m https://10.0.2.2:4242/{0} -p '{1}' -d -q -mp &>> {2} &".format(file_mpd, playback, file_out)
    #    else:
    #        #-n : limit segment count
    #        cmd = "nice -n -10 python3 src/AStream/dist/client/dash_client.py -m https://10.0.2.2:4242/{0} -n 30 -p '{1}' -q -mp &>> {2}".format(file_mpd, playback, file_out)
    #        #file_mpd = '4k60fps.webm'
    #        #cmd = "nice -n -10 python3 src/AStream/dist/client/bulk_transfer.py -m https://10.0.2.2:4242/{0} -p '{1}' -q -mp >> {2} &".format(file_mpd, playback, file_out)
#
    #    print(cmd)
    #    net[ 'client' ].cmd(cmd)
#
    #net['server'].cmd("kill -9 {0}".format(server_pid))
    print('Finishing experiment')

    #end = datetime.now()
    #print(divmod((end - start).total_seconds(), 60))

    #CLI( net )

    net.stop()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Mode execute Mininet')
    parser.add_argument('--background', '-b',
                   metavar='background',
                   type=int,
                   default=0,
                   help='execute with background or not')

    parser.add_argument('--number_client', '-nm',
                   metavar='number_client',
                   type=int,
                   default=2,
                   help='the number of interface client')

    parser.add_argument('--download', '-d',
                   metavar='download',
                   type=bool,
                   default=False,
                   help="Keep the video files after playback")

    parser.add_argument('--playback', '-p',
                   metavar='playback',
                   type=str,
                   default='basic',
                   help="Playback type (basic, sara, netflix, or all)")

    parser.add_argument('--times', '-t',
                        metavar='times',
                        type=str,
                        default=1,
                        help="Number of types to restart dash server")


    # Execute the parse_args() method
    args                       = parser.parse_args()
    with_background            = args.background
    number_of_interface_client = args.number_client
    download                   = args.download
    playback                   = args.playback
    n_times = args.times

    # if len(sys.argv) > 1:
    #     with_background = int(sys.argv[1])
    # if len(sys.argv) > 2:
    #     number_of_interface_client = int(sys.argv[2])

    setLogLevel( 'info' )
    run()

# tc qdisc del dev r4-eth0 root

# src/dash/caddy/caddy -conf /home/mininet/Caddyfile -quic -mp
# python src/AStream/dist/client/dash_client.py -m https://10.0.1.2:4242/output_dash.mpd -p 'basic' -q -mp >> out
# sudo mn --custom build_mininet_router3.py --topo networkTopo --controller=remote,ip=127.0.0.1 --link=tc -x

# ITGSend -T UDP -a 127.0.0.1 -l -e 1000 -t 200000 -l sender.log -x receiver.log -B E 150 V 1.5 100

# ITGSend -T UDP -a 10.0.5.2 -l -e 1000 -t 200000 -B E 150 V 1.5 100
# ITGSend -T UDP -a 10.0.7.2 -l -e 1000 -t 200000 -B E 150 V 1.5 100

# ITGSend -T TCP -a 10.0.6.2 -m RTTM -E 150 -e 1000 -t 200000
# ITGSend -T TCP -a 10.0.8.2 -m RTTM -E 150 -e 1000 -t 200000