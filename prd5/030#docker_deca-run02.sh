DECA_gitlocal_root="/home/lxc_media/#RTU/DEKA2"
 
docker run --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$DECA_gitlocal_root:/deca" \
  deca-perfect:latest \
  python demos/multi_image_deca2.py -i TestSamples/examples/input/ \
    -o TestSamples/examples/results/multi_image --strategy identity --blend-textures