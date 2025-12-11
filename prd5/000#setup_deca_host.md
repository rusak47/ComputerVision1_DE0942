# Installing DECA on host (failed, but smth may be useful, like old api fixes)

`Super important! create project in path with no spaces and etc. otherwise everything will be broken`

python -c "from decalib.utils import renderer; renderer.set\_rasterizer('standard'); print('compiled successfully')"

<br />

\$ python3 -m venv deca

\$ source deca/bin/activate

\$ pip install torch torchvision --index-url <https://download.pytorch.org/whl/cu121>

<br />

\$ pip install --upgrade pip setuptools wheel

\$ pip install git+<https://github.com/mattloper/chumpy.git> --no-build-isolation

\$ pip install "numpy<2.0" opencv-python pillow PyYAML tqdm scikit-image trimesh pyrender face-alignment yacs kornia==0.6.12 fvcore iopath

`smth for cores - install before pytorch build`

\$ pip install ninja

`The --no-build-isolation flag makes the build process use your current environment (where torch, torchvision, numpy, etc. are already installed), so it no longer fails with “No module named 'torch'”.`

\$ MAX\_JOBS=\$(nproc) pip install "git+<https://github.com/facebookresearch/pytorch3d.git>" --no-build-isolation -v

<br />

\$ git clone <https://github.com/yfeng95/DECA.git>

\$ cd DECA

<br />

~~`Finally install DECA`~~

~~\$ pip install -e .~~ <- nothing to install

\$ python -c "from decalib.deca import DECA; print('DECA IS FULLY WORKING RIGHT NOW')"

`Instead export path to lib`

export PYTHONPATH="/home/lxc\_media/#RTU/Computer Vision(1) (DE0942)/prd5/deca/DECA:\$PYTHONPATH"

<br />

\$ python -c "from deca import DECA; print('Works!')"

\$ python -c " \\

&#x20;       import torch; print(f'PyTorch: {torch.\_\_version\_\_} | CUDA: {torch.cuda.is\_available()}')\\

&#x20;       import pytorch3d; print(f'PyTorch3D: {pytorch3d.\_\_version\_\_}')\\

&#x20;       from deca import DECA; print('✅ DECA ready – test with demos!')"

<br />

`fix old version enum`

\$ sed -i 's/LandmarksType.\_2D/LandmarksType.TWO\_D/g' decalib/datasets/detectors.py

<br />

`Important to run before testing (registration needed)`

\$ bash fetch\_data.sh

\$ mv ./data/FLAME2020/FLAME2020/\* ./data/ 2>/dev/null ||/true

\$ rm -rf ./data/FLAME2020

\$ wget -O data/FLAME\_albedo\_from\_BFM.npz "<https://huggingface.co/camenduru/TalkingHead/resolve/main/FLAME_albedo_from_BFM.npz>"

<br />

\$ rm data/deca\_model.tar

\$ wget -O data/deca\_model.tar "<https://huggingface.co/camenduru/show/resolve/d95d9e7901b981d744390e052e05432499d3106c/models/models_deca/data/deca_model.tar>"

<br />

\$ sudo pacman -S cuda cuda-tools

\$ source /etc/profile <- why profile

\$ export CUDA\_HOME=/opt/cuda

<br />

\$ sed -i 's/-ccbin=\$\$(which gcc-7)/-ccbin=gcc/g' decalib/utils/renderer.py

\$ sed -i 's/-std=c++14/-std=c++17/g' decalib/utils/renderer.py

<br />

\$ python demos/demo\_reconstruct.py -i TestSamples/ --saveVis True --saveKpt True --saveDepth True --saveObj True --saveMat True --saveImages True --useTex True --extractTex True
