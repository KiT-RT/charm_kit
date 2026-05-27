#!/usr/bin/env bash
set -euo pipefail

KITRT_REPO_URL="${KITRT_REPO_URL:-https://github.com/KiT-RT/kitrt_code.git}"
KITRT_CONTAINER_BUILD="${KITRT_CONTAINER_BUILD:-auto}"

detect_container_runtime() {
    if [ -n "${KITRT_CONTAINER_RUNTIME:-}" ]; then
        echo "${KITRT_CONTAINER_RUNTIME}"
    elif command -v apptainer >/dev/null 2>&1; then
        echo "apptainer"
    elif command -v singularity >/dev/null 2>&1; then
        echo "singularity"
    else
        echo "ERROR: install Apptainer or Singularity, or set KITRT_CONTAINER_RUNTIME." >&2
        return 1
    fi
}

CONTAINER_RUNTIME="$(detect_container_runtime)"

has_cuda_gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 1
    fi
    nvidia-smi -L >/dev/null 2>&1
}

if [ -d kitrt_code/.git ]; then
    git -C kitrt_code remote set-url origin "${KITRT_REPO_URL}"
else
    git clone "${KITRT_REPO_URL}" kitrt_code
fi

cd kitrt_code
git remote set-url origin "${KITRT_REPO_URL}"
git submodule update --init --recursive

cd tools/singularity
chmod +x build_container.sh install_kitrt_singularity.sh install_kitrt_singularity_cuda.sh

ensure_image() {
    local mode="$1"
    local image="$2"
    local uri="${3:-}"

    if [ -f "${image}" ]; then
        echo "Using existing ${image}."
        return 0
    fi

    if [ -n "${uri}" ]; then
        "${CONTAINER_RUNTIME}" pull "${image}" "${uri}"
        return 0
    fi

    case "${KITRT_CONTAINER_BUILD}" in
        auto|sudo)
            echo "Building ${image} with sudo. Set KITRT_CONTAINER_BUILD=skip or fakeroot on rootless clusters."
            sudo "${CONTAINER_RUNTIME}" build "${image}" "kit_rt${mode}.def"
            ;;
        fakeroot)
            "${CONTAINER_RUNTIME}" build --fakeroot "${image}" "kit_rt${mode}.def"
            ;;
        skip)
            echo "ERROR: ${image} is missing. Provide it, set a pull URI, or allow a container build." >&2
            return 1
            ;;
        *)
            echo "ERROR: KITRT_CONTAINER_BUILD must be auto, sudo, fakeroot, or skip." >&2
            return 1
            ;;
    esac
}

ensure_image "" "kit_rt.sif" "${KITRT_CPU_IMAGE_URI:-}"
"${CONTAINER_RUNTIME}" exec kit_rt.sif ./install_kitrt_singularity.sh

if has_cuda_gpu; then
    echo "CUDA GPU detected. Preparing CUDA container and KiT-RT binary."
    ensure_image "_MPI_cuda" "kit_rt_MPI_cuda.sif" "${KITRT_CUDA_IMAGE_URI:-}"
    "${CONTAINER_RUNTIME}" exec --nv kit_rt_MPI_cuda.sif ./install_kitrt_singularity_cuda.sh
else
    echo "No CUDA GPU detected. Skipping CUDA container and CUDA build."
fi

cd ../../../
