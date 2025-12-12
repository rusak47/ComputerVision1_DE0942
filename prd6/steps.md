sudo pacman -S cuda base-devel cmake openexr libxi glfw openmp libxinerama libxcursor

download optix 
sudo mkdir /opt/nvidia-optix

sudo chown user:users /opt/nvidia-optix
sh NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64.sh --prefix=/opt/nvidia-optix --skip-license

git clone --recursive https://github.com/nvlabs/instant-ngp
cd instant-ngp

!!!build fail gcc14 incompability
------- grok
export PATH=/opt/cuda/bin:$PATH
export LD_LIBRARY_PATH=/opt/cuda/lib:$LD_LIBRARY_PATH
export OptiX_INSTALL_DIR=/opt/nvidia-optix

yay -S gcc13 
~1.1h to build...

------
#  environment variable for the GPU 3080
TCNN_CUDA_ARCHITECTURES=86


#set env vars before CMake.
export CC=gcc-13 CXX=g++-13 TCNN_CUDA_ARCHITECTURES=86

rm -rf build
#CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j

!!!build fail internal nvcc gcc mismatch
--------- export and patch
export CUDAHOSTCXX=/usr/bin/g++-13
export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++-13'

# Backup the file
sudo cp /opt/cuda/targets/x86_64-linux/include/crt/math_functions.h /opt/cuda/targets/x86_64-linux/include/crt/math_functions.h.bak

# Apply patches (edit with sed; targets lines ~6046 & ~6072)
sudo sed -i 's/__func__(double rsqrt(double a));/__func__(double rsqrt(double a)) noexcept(true);/' /opt/cuda/targets/x86_64-linux/include/crt/math_functions.h
sudo sed -i 's/__func__(float rsqrtf(float a));/__func__(float rsqrtf(float a)) noexcept(true);/' /opt/cuda/targets/x86_64-linux/include/crt/math_functions.h

# Rebuild with your current GCC 13 setup (no changes needed)
CC=gcc-13 CXX=g++-13 cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j$(nproc)
-------
export CC=gcc-13 CXX=g++-13 TCNN_CUDA_ARCHITECTURES=86
export CUDAHOSTCXX=/usr/bin/g++-13
export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++-13'
#builds like a charm in few minutes
rm -rf build
cmake . -B build   -DCMAKE_BUILD_TYPE=RelWithDebInfo   -DTCNN_CUDA_ARCHITECTURES=86 
cmake --build build --config RelWithDebInfo -j


# -DTCNN_HALF_PRECISION=1 → tiny-cuda-nn compiles all fp16 kernels (recommended on RTX 3080 Laptop). otherwise rendering will fail
### patch for 
sed -i '/^#pragma once/a #define TCNN_HALF_PRECISION 1' dependencies/tiny-cuda-nn/include/tiny-cuda-nn/common.h
##### revert when needed
Why this works: Adds noexcept(true) to the conflicting __func__ declarations, matching glibc/GCC 13 expectations. Based on Gentoo's upstream patch for glibc 2.41+ (Arch uses 2.41 as of Dec 2025).
Revert: sudo mv /opt/cuda/targets/x86_64-linux/include/crt/math_functions.h.bak /opt/cuda/targets/x86_64-linux/include/crt/math_functions.h
Pacman Note: This survives pacman -Syu cuda (headers are in /opt/cuda/targets/), but re-patch if needed.
######### upgrade nvidia driver to meet toolkit version <- doesnt work in my case
sudo pacman -S nvidia-open-dkms lib32-nvidia-utils nvidia-utils nvidia-settings
WARNING: updating nvidia-utils version (570.133.07 -> 580.105.08) in /etc/cdi/nvidia.yaml using plain string substitution.
 -> If you meet problems, run the following command to regenerate the CDI file:
    nvidia-ctk cdi generate --output="/etc/cdi/nvidia.yaml"

###### >>> instead downgrade cuda toolkit to meet nvidia driver version
sudo downgrade cuda cudnn
extra/cuda   12.8.1-3
cudnn        9.11.0.98-3
then rebuild 
######### test
./build/instant-ngp 
11:17:22 SUCCESS  Initialized CUDA 12.8. Active GPU is #0: NVIDIA GeForce RTX 3080 Laptop GPU [86]
Can't open bumblebee display.
11:17:22 WARNING  Vulkan instance validation layer is not available. Vulkan errors will be difficult to diagnose.
11:17:22 SUCCESS  Initialized Vulkan and NGX on GPU #0: NVIDIA GeForce RTX 3080 Laptop GPU
11:17:22 SUCCESS  Initialized OpenGL version 4.6 (Compatibility Profile) Mesa 25.0.3-arch1.1

to check
(If you ever want the absolute maximum speed on your 3080 Laptop, you can even add -DTCNN_SHMEM_SIZE=49152 for the bigger shared-memory kernels, but the default is already excellent.)
##########

