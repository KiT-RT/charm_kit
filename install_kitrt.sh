#!/usr/bin/env bash
set -euo pipefail

KITRT_REPO_URL="${KITRT_REPO_URL:-https://github.com/KiT-RT/kitrt_code.git}"

has_cuda_gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 1
    fi
    nvidia-smi -L >/dev/null 2>&1
}

has_rocm_gpu() {
    if command -v rocm-smi >/dev/null 2>&1; then
        rocm-smi --showid 2>/dev/null | grep -q "^GPU\\["
        return
    fi

    if command -v rocminfo >/dev/null 2>&1; then
        rocminfo 2>/dev/null | grep -qi "gfx"
        return
    fi

    return 1
}

# clone KiT-RT once and reuse the existing checkout on subsequent runs
if [ -d "kitrt_code/.git" ]; then
    echo "Existing kitrt_code checkout detected. Skipping clone + checkout."
elif [ -d "kitrt_code" ]; then
    echo "Directory kitrt_code exists but is not a git repository. Please remove or rename it first." >&2
    exit 1
else
    git clone "${KITRT_REPO_URL}" kitrt_code
fi

# go to kitrt_code directory
cd kitrt_code

# keep origin synchronized with configured upstream and load submodules
git remote set-url origin "${KITRT_REPO_URL}"
git submodule update --init --recursive

# navigate to directory where the singularity scripts are located
cd tools/singularity
chmod +x \
    build_container.sh \
    install_kitrt_singularity.sh \
    install_kitrt_singularity_cuda.sh \
    install_kitrt_singularity_rocm.sh

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

# optionally build and compile ROCm KiT-RT if a ROCm GPU is present
if has_rocm_gpu; then
    if [ -f "kit_rt_MPI_rocm72.def" ]; then
        echo "ROCm GPU detected. Building ROCm singularity container and ROCm KiT-RT binary."
        sudo ./build_container.sh rocm
        singularity exec --rocm kit_rt_MPI_rocm72.sif ./install_kitrt_singularity_rocm.sh
    else
        echo "ROCm GPU detected, but kit_rt_MPI_rocm72.def was not found. Skipping ROCm build."
    fi
else
    echo "No ROCm GPU detected. Skipping ROCm singularity container and ROCm build."
fi

# go back to CharmKiT repo root
cd ../../../
