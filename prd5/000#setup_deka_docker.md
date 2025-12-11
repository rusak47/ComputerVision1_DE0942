\#install missing packages

\$ sudo pacman -S nvidia-container-toolkit

<br />

\$ pacman -Q | grep -E 'nvidia|cuda|cudnn|nccl'

cuda 13.0.2-1
cuda-tools 13.0.2-1

cudnn 9.16.0.29-1

lib32-nvidia-utils 570.133.07-1

libnvidia-container 1.18.0-1

nvidia-container-toolkit 1.18.0-1

nvidia-open-dkms 570.133.07-1

nvidia-prime 1.0-5

nvidia-utils 570.133.07-1

opencl-nvidia 570.133.07-1

<br />

\#configure nvidia for docker access and run a test container

\$ sudo nvidia-ctk runtime configure --runtime=docker

\$ docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

there should be a standard output

<br />

\#build deka docker container

\$ docker build -t deca-jupyter -f Dockerfile .

