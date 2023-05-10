#entry copied from https://github.com/rixsilverith/mininet-docker/tree/main
FROM ubuntu:22.04

USER root
WORKDIR /root

COPY ENTRYPOINT.sh /

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y --no-install-recommends \
    curl \
    dnsutils \
    ifupdown \
    iproute2 \
    iptables \
    iputils-ping \
    mininet \
    net-tools \
    openvswitch-switch \
    openvswitch-testcontroller \
    tcpdump \
    vim \
    x11-xserver-utils \
    xterm \
    python3 \
    python3-pip \
	git \
	sudo\
 && rm -rf /var/lib/apt/lists/* \
 && touch /etc/network/interfaces \
 && chmod +x /ENTRYPOINT.sh
 
#ensure we can connect to the internet
RUN sysctl -w net.ipv4.ip_forward=1

EXPOSE 6633 6653 6640 22

#go1.20.4
#go1.16.15
#COPY mininet/ mininet/

#add mininet user
RUN useradd -rm -d /home/mininet -s /bin/bash -g root -G sudo -u 1001 mininet
USER mininet
WORKDIR /home/mininet

#Install go
RUN curl -OL https://golang.org/dl/go1.12.2.linux-amd64.tar.gz
RUN tar -C /usr/local -xzf go1.12.2.linux-amd64.tar.gz
ENV PATH="$PATH:/usr/local/go/bin"


ENTRYPOINT ["/ENTRYPOINT.sh"]