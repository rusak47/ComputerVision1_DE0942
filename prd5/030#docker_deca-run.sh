DECA_gitlocal_root="/home/lxc_media/#RTU/DEKA2"
 
docker run --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$DECA_gitlocal_root:/deca" \
  deca-perfect:latest \
  python demos/demo_reconstruct.py -i TestSamples/examples/input/ \
    --saveVis True --saveObj True --saveImages True --useTex True --extractTex True