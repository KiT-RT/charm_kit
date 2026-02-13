import subprocess
import os
import time
from src.general_utils import get_user_job_count


def _get_cuda_visible_device_count():
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        return None

    parsed = [value.strip() for value in visible_devices.split(",") if value.strip()]
    parsed = [value for value in parsed if value != "-1"]
    return len(parsed)


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


def _resolve_cuda_mpi_ranks(quiet=False):
    override = os.environ.get("KITRT_CUDA_MPI_RANKS")
    if override is not None:
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
    # Path to the C++ executable
    if use_cuda:
        mpi_ranks = _resolve_cuda_mpi_ranks(quiet=quiet)
        singularity_command = [
            "singularity",
            "exec",
            "--nv",
            "kitrt_code/tools/singularity/kit_rt_MPI_cuda.sif",
            "mpirun",
            "-np",
            mpi_ranks,
            "./kitrt_code/build_singularity_cuda/KiT-RT",
            config_file,
        ]
    else:
        singularity_command = [
            "singularity",
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
