#imageid=$(docker build -t mininet-rl -q .)
docker run --rm -it $(docker build -t mininet-rl -q .)