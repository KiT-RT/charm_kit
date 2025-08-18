# import umbridge
import numpy as np

from src.config_utils import read_username_from_config
from src.models.lattice import get_qois_col_names, model

from src.simulation_utils import execute_slurm_scripts, wait_for_slurm_jobs
from src.general_utils import (
    create_lattice_samples_from_param_range,
    load_lattice_samples_from_npz,
    delete_slurm_scripts,
)

# url = "http://localhost:4243"
# model = umbridge.HTTPModel(url, "forward")

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process some flags for HPC and mesh operations."
    )

    # Add arguments
    parser.add_argument(
        "--no-hpc", action="store_true", help="Flag when using HPC cluster"
    )
    parser.add_argument(
        "--load-from-npz", action="store_true", help="Flag to load from NPZ file"
    )
    parser.add_argument(
        "--no-singularity-hpc",
        action="store_true",
        help="Flag to use Singularity on HPC",
    )
    parser.add_argument(
        "--rectangular-mesh",
        action="store_true",
        help="Flag for using rectangular mesh",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(f"HPC mode = { args.use_slurm}")
    print(f"Load from npz = {args.load_from_npz}")
    print(f"HPC with singularity = { args.use_singularity}")

    hpc_operation = args.use_slurm  # Flag when using HPC cluster
    load_from_npz = args.load_from_npz
    singularity_hpc = args.use_singularity

    # print(f"Use rectangular_mesh = {args.rectangular_mesh}")
    rectangular_mesh = False

    # Define parameter ranges
    # characteristic length of the cells
    parameter_range_n_cell = [
        # 0.05,
        # 0.025,
        # 0.01,
        # 0.0075,
        # 0.005,
        0.0025,
        # 0.001,
        # 0.00075,
        # 0.0005,
    ]
    #    0.0075,
    #    0.005,
    #    0.0025,
    #    0.001,
    # ]
    # GAUSS LEGENDRE  2D quadrature order (MUST BE EVEN)

    parameter_range_quad_order = [
        4
    ]  # [4, 8, 12, 16, 20, 24, 28,32,40,48,56,64,72,80]  # [4, 8, 12, 16, 20, 24, 28]
    #    20,
    #    30,
    #    40,
    #    50,
    #    60,
    # ]  # GAUSS LEGENDRE 2D quadrature

    parameter_range_abs_blue = [
        10
    ]  # [0, 5, 10, 50, 100]  # Prescribed range for LATTICE_DSGN_ABSORPTION_BLUE
    parameter_range_scatter_white = [
        1
    ]  # [0, 0.5, 1, 5, 10]  # Prescribed range for LATTICE_DSGN_ABSORPTION_BLUE

    if load_from_npz:  # TODO
        design_params, design_param_names = load_lattice_samples_from_npz(
            "sampling/pilot-study-samples-hohlraum-05-29-24.npz"
        )
        exit("TODO")
    else:
        design_params, design_param_names = create_lattice_samples_from_param_range(
            parameter_range_n_cell,
            parameter_range_quad_order,
            parameter_range_abs_blue,
            parameter_range_scatter_white,
        )

    if hpc_operation:
        print("==== Execute HPC version ====")
        directory = "./benchmarks/lattice/slurm_scripts/"

        delete_slurm_scripts(directory)  # delete existing slurm files for hohlraum
        call_models(
            design_params,
            hpc_operation_count=1,
            singularity_hpc=singularity_hpc,
            rectangular_mesh=rectangular_mesh,
        )

        user = read_username_from_config("./slurm_config.txt")
        if user:
            print("Executing slurm scripts with user " + user)
            execute_slurm_scripts(directory, user)
            wait_for_slurm_jobs(user=user, sleep_interval=10)
        else:
            print("Username could not be read from config file.")

        qois = call_models(design_params, hpc_operation_count=2)
    else:
        qois = call_models(
            design_params,
            hpc_operation_count=0,
            rectangular_mesh=rectangular_mesh,
            singularity_hpc=singularity_hpc,
        )

    print("design parameter matrix")
    print(design_param_names)
    print(design_params)
    print("quantities of interest:")
    print(get_qois_col_names())
    print(qois)
    np.savez(
        "benchmarks/lattice/sn_study_lattice.npz",
        qois=qois,
        design_params=design_params,
        qoi_column_names=get_qois_col_names(),
        design_param_column_names=design_param_names,
    )

    print("======== Finished ===========")
    return 0


def call_models(
    design_params, hpc_operation_count, singularity_hpc=True, rectangular_mesh=False
):
    qois = []
    for column in design_params:
        input = column.tolist()
        print(input)
        input.append(hpc_operation_count)
        input.append(singularity_hpc)
        input.append(rectangular_mesh)

        res = model([input])
        qois.append(res[0])

    return np.array(qois)


if __name__ == "__main__":
    main()
