#!/usr/bin/env bash
set -euo pipefail

KITRT_REPO_URL="${KITRT_REPO_URL:-git@github.com:KiT-RT/kitrt_code.git}"

has_cuda_gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 1
    fi
    nvidia-smi -L >/dev/null 2>&1
}

cd KiT-RT
git remote set-url origin "${KITRT_REPO_URL}"
git fetch origin

default_branch="$(git symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null | sed 's#^origin/##' || true)"
if [ -z "${default_branch}" ]; then
    default_branch="$(git remote show origin | sed -n '/HEAD branch/s/.*: //p' | head -n 1)"
fi
if [ -z "${default_branch}" ]; then
    default_branch="$(git rev-parse --abbrev-ref HEAD)"
fi

git checkout "${default_branch}"
git pull --ff-only origin "${default_branch}"
git submodule update --init --recursive

cd tools/singularity
chmod +x install_kitrt_singularity.sh install_kitrt_singularity_cuda.sh

singularity exec kit_rt.sif ./install_kitrt_singularity.sh

if has_cuda_gpu && [ -f kit_rt_MPI_cuda.sif ]; then
    singularity exec --nv kit_rt_MPI_cuda.sif ./install_kitrt_singularity_cuda.sh
fi

cd ../../../
