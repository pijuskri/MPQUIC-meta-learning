sudo rsync -azP -e 'ssh -p 2222' --exclude={'*/venv/*','*/proxy_module/*','*.so'} mpquic-sbd mininet@172.23.160.1:/home/mininet/Workspace/
bash ./ssh_exec.sh "~/Workspace/mpquic-sbd/build.sh"