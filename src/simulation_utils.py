import glob
import os
import shutil
import subprocess
import time
from src.general_utils import get_user_job_count


def _get_visible_device_count(env_var_name):
    visible_devices = os.environ.get(env_var_name)
    if visible_devices is None:
        return None

    parsed = [value.strip() for value in visible_devices.split(",") if value.strip()]
    parsed = [value for value in parsed if value != "-1"]
    return len(parsed)


def _get_cuda_visible_device_count():
    return _get_visible_device_count("CUDA_VISIBLE_DEVICES")


def _get_rocm_visible_device_count():
    for env_var_name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        visible_count = _get_visible_device_count(env_var_name)
        if visible_count is not None:
            return visible_count
    return None


def _has_cuda_gpu():
    visible_count = _get_cuda_visible_device_count()
    if visible_count is not None:
        return visible_count >= 1
    return _query_nvidia_smi_gpu_count() >= 1


def _has_rocm_gpu():
    visible_count = _get_rocm_visible_device_count()
    if visible_count is not None:
        return visible_count >= 1
    return _query_rocm_smi_gpu_count() >= 1


def _query_nvidia_smi_gpu_count():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        return 0

    if result.returncode != 0:
        return 0

    return len([line for line in (result.stdout or "").splitlines() if line.strip()])


def _query_rocm_smi_gpu_count():
    try:
        result = subprocess.run(
            ["rocm-smi", "--showid"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        return 0

    if result.returncode != 0:
        return 0

    gpu_ids = set()
    for line in (result.stdout or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("GPU["):
            continue

        closing_bracket = stripped.find("]")
        if closing_bracket <= 4:
            continue

        gpu_id = stripped[4:closing_bracket]
        if gpu_id.isdigit():
            gpu_ids.add(gpu_id)

    return len(gpu_ids)


def _resolve_gpu_mpi_ranks_override():
    override = os.environ.get("KITRT_CUDA_MPI_RANKS")
    if override is None:
        return None

    try:
        rank_count = int(override)
    except ValueError as e:
        raise RuntimeError(
            "Invalid KITRT_CUDA_MPI_RANKS value. Expected positive integer, "
            f"got: {override!r}"
        ) from e
    if rank_count < 1:
        raise RuntimeError(
            "Invalid KITRT_CUDA_MPI_RANKS value. Expected >= 1, "
            f"got: {rank_count}"
        )
    return str(rank_count)


def _resolve_cuda_mpi_ranks(quiet=False):
    override = _resolve_gpu_mpi_ranks_override()
    if override is not None:
        return override

    visible_count = _get_cuda_visible_device_count()
    if visible_count is not None:
        if visible_count >= 1:
            return str(visible_count)
        if not quiet:
            print("CUDA_VISIBLE_DEVICES is empty; falling back to 1 MPI rank.")
        return "1"

    detected_gpu_count = _query_nvidia_smi_gpu_count()
    if detected_gpu_count >= 1:
        return str(detected_gpu_count)

    if not quiet:
        print("Could not detect available GPUs; falling back to 1 MPI rank.")
    return "1"


def _resolve_rocm_mpi_ranks(quiet=False):
    override = _resolve_gpu_mpi_ranks_override()
    if override is not None:
        return override

    visible_count = _get_rocm_visible_device_count()
    if visible_count is not None:
        if visible_count >= 1:
            return str(visible_count)
        if not quiet:
            print(
                "HIP/ROCR visible-device mask is empty; falling back to 1 MPI rank."
            )
        return "1"

    detected_gpu_count = _query_rocm_smi_gpu_count()
    if detected_gpu_count >= 1:
        return str(detected_gpu_count)

    if not quiet:
        print("Could not detect available ROCm GPUs; falling back to 1 MPI rank.")
    return "1"


def _is_rocm_installed():
    return (
        shutil.which("rocm-smi") is not None
        or shutil.which("rocminfo") is not None
        or os.path.isdir("/opt/rocm")
    )


def _find_rocm_container_image():
    preferred_images = [
        "kitrt_code/tools/singularity/kit_rt_MPI_rocm72.sif",
        "kitrt_code/tools/singularity/kit_rt_MPI_rocm.sif",
    ]
    for image_path in preferred_images:
        if os.path.exists(image_path):
            return image_path

    rocm_images = sorted(
        glob.glob("kitrt_code/tools/singularity/kit_rt_MPI_rocm*.sif")
    )
    if rocm_images:
        return rocm_images[0]
    return None


def _find_rocm_executable():
    preferred_executables = [
        "./kitrt_code/build_singularity_rocm72/KiT-RT",
        "./kitrt_code/build_singularity_rocm/KiT-RT",
    ]
    for executable_path in preferred_executables:
        if os.path.exists(executable_path):
            return executable_path

    rocm_executables = sorted(
        glob.glob("./kitrt_code/build_singularity_rocm*/KiT-RT")
    )
    if rocm_executables:
        return rocm_executables[0]
    return None


def _resolve_container_runtime():
    runtime_override = os.environ.get("KITRT_CONTAINER_RUNTIME")
    if runtime_override:
        runtime = runtime_override.strip().lower()
        if runtime not in ("apptainer", "singularity"):
            raise RuntimeError(
                "Invalid KITRT_CONTAINER_RUNTIME value. Expected "
                "'apptainer' or 'singularity', "
                f"got: {runtime_override!r}"
            )
        if shutil.which(runtime) is None:
            raise RuntimeError(
                f"KITRT_CONTAINER_RUNTIME is set to {runtime!r}, "
                "but that executable was not found in PATH."
            )
        return runtime

    if shutil.which("apptainer") is not None:
        return "apptainer"
    if shutil.which("singularity") is not None:
        return "singularity"

    raise RuntimeError(
        "Containerized KiT-RT run failed: neither 'apptainer' nor "
        "'singularity' was found in PATH."
    )


def _run_and_raise(command, mode_label, quiet=False):
    try:
        if quiet:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        else:
            result = subprocess.run(command)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"{mode_label} run failed: executable not found: {command[0]}"
        ) from e

    if result.returncode != 0:
        stderr = ""
        if quiet:
            stderr = (result.stderr or "").strip()
        hint = ""
        if "error while loading shared libraries" in stderr:
            hint = (
                " Hint: missing system libraries in local mode. "
                "Use --singularity (or --cuda) to run inside the container."
            )
        if not stderr:
            stderr = (
                "See solver output above."
                if not quiet
                else "No stderr captured."
            )
        raise RuntimeError(
            f"{mode_label} run failed with return code {result.returncode}. "
            f"Command: {' '.join(command)}. "
            f"Stderr: {stderr}{hint}"
        )


def run_cpp_simulation(config_file, quiet=False):
    # Path to the C++ executable
    print("here")
    current_path = os.getcwd()

    # Print the current path
    print(f"The current working directory is: {current_path}")
    print(config_file)
    cpp_executable_path = "./kitrt_code/build/KiT-RT"  # mpirun -np 4

    # Command to run the C++ executable with the provided config file
    command = [cpp_executable_path, config_file]

    print(command)
    _run_and_raise(command, "Local KiT-RT", quiet=quiet)
    print("C++ simulation completed successfully.")


def run_cpp_simulation_containerized(config_file, use_cuda=False, quiet=False):
    container_runtime = _resolve_container_runtime()

    # Path to the C++ executable
    if use_cuda:
        # Keep the existing public flag, but select CUDA or ROCm at runtime.
        if _has_cuda_gpu():
            mpi_ranks = _resolve_cuda_mpi_ranks(quiet=quiet)
            singularity_command = [
                container_runtime,
                "exec",
                "--nv",
                "kitrt_code/tools/singularity/kit_rt_MPI_cuda.sif",
                "mpirun",
                "-np",
                mpi_ranks,
                "./kitrt_code/build_singularity_cuda/KiT-RT",
                config_file,
            ]
        elif _is_rocm_installed() and _has_rocm_gpu():
            rocm_image = _find_rocm_container_image()
            if rocm_image is None:
                raise RuntimeError(
                    "ROCm runtime detected, but no KiT-RT ROCm Singularity image was found "
                    "under kitrt_code/tools/singularity/ (expected kit_rt_MPI_rocm*.sif)."
                )

            rocm_executable = _find_rocm_executable()
            if rocm_executable is None:
                raise RuntimeError(
                    "ROCm runtime detected, but no KiT-RT ROCm executable was found "
                    "under kitrt_code/build_singularity_rocm*/KiT-RT."
                )

            if not quiet:
                print(
                    "CUDA GPUs not detected; using ROCm KiT-RT container and executable."
                )

            mpi_ranks = _resolve_rocm_mpi_ranks(quiet=quiet)
            singularity_command = [
                container_runtime,
                "exec",
                "--rocm",
                rocm_image,
                "mpirun",
                "-np",
                mpi_ranks,
                rocm_executable,
                config_file,
            ]
        else:
            if not quiet:
                print(
                    "CUDA GPUs were not detected and no ROCm GPU fallback is available; "
                    "running CPU KiT-RT container path."
                )
            singularity_command = [
                container_runtime,
                "exec",
                "kitrt_code/tools/singularity/kit_rt.sif",
                "./kitrt_code/build_singularity/KiT-RT",
                config_file,
            ]
    else:
        singularity_command = [
            container_runtime,
            "exec",
            "kitrt_code/tools/singularity/kit_rt.sif",
            "./kitrt_code/build_singularity/KiT-RT",
            config_file,
        ]

    # Command to run the C++ executable with the provided config file

    _run_and_raise(singularity_command, "Containerized KiT-RT", quiet=quiet)
    print("C++ simulation completed successfully.")


def execute_slurm_scripts(directory, user, max_jobs=60, sleep_time=30):
    """
    Execute all SLURM scripts in the specified directory.
    If the number of jobs in the queue for the user is 10 or more, wait and sleep for 30 seconds.
    """
    # Get the list of SLURM scripts in the directory
    slurm_scripts = [f for f in os.listdir(directory) if f.endswith(".sh")]

    #print(slurm_scripts)

    for script in slurm_scripts:
        script_path = os.path.join(directory, script)

        # Check the number of jobs in the queue for the user
        while get_user_job_count(user) >= max_jobs:
            print(
                f"User has {max_jobs} or more jobs in the queue. Waiting for {sleep_time} seconds..."
            )
            time.sleep(sleep_time)

        # Execute the SLURM script
        try:
            result = subprocess.run(
                ["sbatch", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            if result.returncode == 0:
                print(f"Successfully submitted {script}")
            else:
                print(f"Failed to submit {script}: {result.stderr}")
        except Exception as e:
            print(f"Error submitting {script}: {e}")


def wait_for_slurm_jobs(user, sleep_interval=30):
    """
    Waits until all SLURM jobs for the specified user are finished.

    Parameters:
    - user (str): The username to check SLURM jobs for.
    - sleep_interval (int): The number of seconds to wait between checks. Default is 30 seconds.
    """
    while True:
        try:
            # Get the list of jobs for the user
            result = subprocess.run(
                ["squeue", "-u", user],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            # Split the result into lines
            lines = result.stdout.strip().split("\n")

            # The first line is the header, so if there are more than 1 lines, there are running jobs
            if len(lines) <= 1:
                print("All SLURM jobs for user '{}' are finished.".format(user))
                break

            # Print the current status
            print("Waiting for SLURM jobs to finish. Current jobs:")
            for line in lines:
                print(line)

            # Wait for the specified interval before checking again
            time.sleep(sleep_interval)

        except subprocess.CalledProcessError as e:
            print("An error occurred while checking SLURM jobs: {}".format(e))
            break
