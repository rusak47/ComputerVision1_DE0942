DECA_gitlocal_root="/home/lxc_media/#RTU/DEKA2"

# Create the folder on the host (once)
# mkdir -p $DECA_gitlocal_root/jupyter-lab

# Get user real IDs
MY_UID=$(id -u)
MY_GID=$(id -g)

# remove previous run instance
docker rm deca-jupyter 

# launch as host user
docker run -d --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$DECA_gitlocal_root:/deca" \
  -p 8888:8888 \
  --user "$MY_UID:$MY_GID" \
  --userns=host \
  --env HOME=/deca/jupyter-lab \
  --workdir /deca/jupyter-lab \
  --name deca-jupyter \
  deca-perfect:latest \
  jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token= \
    --NotebookApp.notebook_dir=/deca/jupyter-lab \
    --ContentsManager.root_dir=/deca/jupyter-lab \
    --ServerApp.root_dir=/deca/jupyter-lab \
    --ServerApp.default_url=/lab

# container may fail after few seconds, so wait a little and output its status
sleep 10 && docker ps -a 
docker logs deca-jupyter
