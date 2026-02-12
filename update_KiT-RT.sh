#!/usr/bin/env bash
set -euo pipefail

has_cuda_gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 1
    fi
    nvidia-smi -L >/dev/null 2>&1
}

cd KiT-RT
git checkout master
git pull origin
cd tools/singularity
chmod +x install_kitrt_singularity.sh install_kitrt_singularity_cuda.sh

singularity exec kit_rt.sif ./install_kitrt_singularity.sh

if has_cuda_gpu && [ -f kit_rt_MPI_cuda.sif ]; then
    singularity exec --nv kit_rt_MPI_cuda.sif ./install_kitrt_singularity_cuda.sh
fi

cd ../../../
