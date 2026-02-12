from src.config_utils import (
    generate_log_filename,
    read_config_file,
    update_parameter,
    write_config_file,
    remove_files,
    update_lattice_mesh_file,
    write_slurm_file,
)
from src.scraping_utils import read_csv_file
from src.simulation_utils import run_cpp_simulation_containerized, run_cpp_simulation
import os
import numpy as np


def model(parameters):
    """
    A function that performs a series of steps including reading a base config file,
    updating parameters based on input values, running a C++ simulation, and processing
    the output data. It takes in parameters and a config dictionary, and returns a list
    of quantities of interest calculated during the simulation.
    """

    absorption_blue_value = parameters[0][0]
    scatter_white_value = parameters[0][1]

    n_cells = parameters[0][2]
    quad_order = int(parameters[0][3])

    hpc_operation = parameters[0][4]
    singularity_hpc = parameters[0][5]
    rectangular_mesh = parameters[0][6]
    use_cuda = bool(parameters[0][7]) if len(parameters[0]) > 7 else False
    if use_cuda:
        singularity_hpc = 1
    if use_cuda and hpc_operation == 1:
        raise ValueError(
            "CUDA mode with SLURM is not supported in this workflow."
        )

    subfolder = "benchmarks/lattice/"
    base_config_file = subfolder + "lattice.cfg"

    # Step 1: Read the base config file
    kitrt_parameters = read_config_file(base_config_file)
    lattice_file_new = update_lattice_mesh_file(
        n_cells, subfolder + "mesh/", rectangular_mesh=rectangular_mesh
    )

    unique_name = f"lattice_abs{absorption_blue_value}_scatter{scatter_white_value}_p{n_cells}_q{quad_order}"
    if hpc_operation == 2:
        if os.path.exists(
            subfolder + "mesh/" + "hohlraum_variable_backup" + unique_name + ".geo"
        ):
            os.remove(
                subfolder + "mesh/" + "hohlraum_variable_backup" + unique_name + ".geo"
            )  # remove backup geo files

    # Step 2: Update kitrt_parameters for the current value of LATTICE_DSGN_ABSORPTION_BLUE
    kitrt_parameters = update_parameter(
        kitrt_parameters,
        key="LATTICE_DSGN_ABSORPTION_BLUE",
        new_value=absorption_blue_value,
    )
    kitrt_parameters = update_parameter(
        kitrt_parameters,
        key="LATTICE_DSGN_SCATTER_WHITE",
        new_value=scatter_white_value,
    )
    kitrt_parameters = update_parameter(
        kitrt_parameters, key="QUAD_ORDER", new_value=quad_order
    )
    kitrt_parameters = update_parameter(
        kitrt_parameters, key="MESH_FILE", new_value="mesh/" + lattice_file_new
    )

    # Step 3: Update LOG_FILE to a unique identifier linked to LATTICE_DSGN_ABSORPTION_BLUE

    kitrt_parameters = update_parameter(
        kitrt_parameters, key="LOG_FILE", new_value=unique_name
    )
    if hpc_operation < 2:
        remove_files(subfolder + kitrt_parameters["LOG_DIR"] + "/" + unique_name)
    kitrt_parameters = update_parameter(
        kitrt_parameters, key="OUTPUT_FILE", new_value=unique_name
    )
    if hpc_operation < 2:
        remove_files(
            subfolder + kitrt_parameters["OUTPUT_DIR"] + "/" + unique_name + ".vtk"
        )

    # Step 4: Write a new config file, named corresponding to LATTICE_DSGN_ABSORPTION_BLUE
    generated_cfg_file = subfolder + unique_name + ".cfg"
    write_config_file(parameters=kitrt_parameters, output_file_path=generated_cfg_file)

    # Step 5: Run the C++ simulation
    if hpc_operation == 0:
        # Step 5: Run the C++ simulation
        if singularity_hpc == 1:
            print("Running lattice simulation with singularity")
            run_cpp_simulation_containerized(generated_cfg_file, use_cuda=use_cuda)
        else:
            print("Running lattice simulation without singularity")
            run_cpp_simulation(generated_cfg_file)
    elif hpc_operation == 1:
        # Write slurm file
        write_slurm_file(
            "benchmarks/lattice/slurm_scripts/",
            unique_name,
            subfolder,
            singularity_hpc,
            use_cuda=use_cuda,
        )

    # Step 6: Read the log file
    if hpc_operation == 0 or hpc_operation == 2:
        log_filename = generate_log_filename(kitrt_parameters)

        if log_filename:
            # Step 7: Read and convert the data from the CSV log file to a DataFrame
            log_data = read_csv_file(subfolder + log_filename + ".csv")
            log_data["LATTICE_DSGN_ABSORPTION_BLUE"] = absorption_blue_value
            log_data["LATTICE_DSGN_SCATTER_WHITE"] = scatter_white_value
            quantities_of_interest = [
                float(log_data["Wall_time_[s]"]),
                float(log_data["Mass"]),
                float(log_data["Cur_absorption"]),
                float(log_data["Total_absorption"]),
                float(log_data["Cur_outflow_P1"]),
                float(log_data["Total_outflow_P1"]),
                float(log_data["Cur_outflow_P2"]),
                float(log_data["Total_outflow_P2"]),
            ]
    else:
        quantities_of_interest = [0] * 8

    return [quantities_of_interest]


def get_qois_col_names():
    return np.array(
        [
            "Wall_time_[s]",
            "Mass",
            "Cur_absorption",
            "Total_absorption",
            "Cur_outflow_P1",
            "Total_outflow_P1",
            "Cur_outflow_P2",
            "Total_outflow_P2",
        ]
    )
