#imageid=$(docker build -t mininet-rl -q .)
#docker run -u 0 --rm -it $(docker build -t mininet-rl -q .)
#docker run -it --rm --privileged -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /lib/modules:/lib/modules $(docker build -t mininet-rl -q .)
#docker run -it --rm --privileged $(docker build -t mininet-rl -q .)
#docker run --name containernet -it --rm --privileged --pid='host' -v /var/run/docker.sock:/var/run/docker.sock $(docker build -t mininet-rl -q .)

docker run -it --rm --privileged -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /lib/modules:/lib/modules -v $(pwd):/data $(docker build -t mininet-rl -q .)