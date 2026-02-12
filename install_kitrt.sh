#!/usr/bin/env bash
set -euo pipefail

KITRT_REPO_URL="${KITRT_REPO_URL:-git@github.com:KiT-RT/kitrt_code.git}"

has_cuda_gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 1
    fi
    nvidia-smi -L >/dev/null 2>&1
}

# clone KiT-RT
git clone "${KITRT_REPO_URL}" kitrt_code

# go to kitrt_code directory
cd kitrt_code

# keep origin synchronized with configured upstream and load submodules
git remote set-url origin "${KITRT_REPO_URL}"
git submodule update --init --recursive

# navigate to directory where the singularity scripts are located
cd tools/singularity
chmod +x build_container.sh install_kitrt_singularity.sh install_kitrt_singularity_cuda.sh

# build CPU singularity container. This requires root privileges.
echo "Building CPU singularity container (sudo required)."
sudo ./build_container.sh cpu

# compile CPU KiT-RT within the singularity container
singularity exec kit_rt.sif ./install_kitrt_singularity.sh

# optionally build and compile CUDA KiT-RT if a CUDA GPU is present
if has_cuda_gpu; then
    echo "CUDA GPU detected. Building CUDA singularity container and CUDA KiT-RT binary."
    sudo ./build_container.sh cuda
    singularity exec --nv kit_rt_MPI_cuda.sif ./install_kitrt_singularity_cuda.sh
else
    echo "No CUDA GPU detected. Skipping CUDA singularity container and CUDA build."
fi

# go back to CharmKiT repo root
cd ../../../
