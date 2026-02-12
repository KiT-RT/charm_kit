import subprocess
import os
import time
from src.general_utils import get_user_job_count


def run_cpp_simulation(config_file):
    # Path to the C++ executable
    print("here")
    current_path = os.getcwd()

    # Print the current path
    print(f"The current working directory is: {current_path}")
    print(config_file)
    cpp_executable_path = "./KiT-RT/build/KiT-RT"  # mpirun -np 4

    # Command to run the C++ executable with the provided config file
    command = [cpp_executable_path, config_file]

    print(command)
    try:
        # Run the C++ executable
        subprocess.run(command, check=True)
        print("C++ simulation completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running C++ simulation. Return code: {e.returncode}")
        # You can handle the error as needed

        # You can handle the error as needed


def run_cpp_simulation_containerized(config_file, use_cuda=False):
    # Path to the C++ executable
    if use_cuda:
        singularity_command = [
            "singularity",
            "exec",
            "--nv",
            "KiT-RT/tools/singularity/kit_rt_MPI_cuda.sif",
            "./KiT-RT/build_singularity_cuda/KiT-RT",
            config_file,
        ]
    else:
        singularity_command = [
            "singularity",
            "exec",
            "KiT-RT/tools/singularity/kit_rt.sif",
            "./KiT-RT/build_singularity/KiT-RT",
            config_file,
        ]

    # Command to run the C++ executable with the provided config file

    try:
        # Run the C++ executable
        subprocess.run(singularity_command, check=True)
        print("C++ simulation completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running C++ simulation. Return code: {e.returncode}")
        # You can handle the error as needed


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
