# import umbridge
import numpy as np
import pandas as pd

from src.config_utils import read_username_from_config
from src.models.lattice import get_qois_col_names, model

from src.simulation_utils import execute_slurm_scripts, wait_for_slurm_jobs
from src.general_utils import (
    create_lattice_samples_from_param_range,
    load_lattice_samples_from_npz,
    delete_slurm_scripts,
)

from src.general_utils import parse_args


def main():
    args = parse_args()
    print(f"HPC mode = { args.use_slurm}")
    print(f"Load from npz = {args.load_from_npz}")
    print(f"HPC with singularity = { args.use_singularity}")
    print(f"CUDA mode = {args.cuda}")

    hpc_operation = args.use_slurm  # Flag when using HPC cluster
    load_from_npz = args.load_from_npz
    use_cuda = args.cuda
    if use_cuda and hpc_operation:
        raise SystemExit(
            "ERROR: --cuda cannot be combined with --use-slurm. "
            "GPU mode is supported only with Singularity in non-SLURM runs."
        )
    singularity_hpc = args.use_singularity or use_cuda
    if use_cuda and not args.use_singularity:
        print("CUDA mode requested; enabling Singularity execution.")

    # print(f"Use rectangular_mesh = {args.rectangular_mesh}")
    rectangular_mesh = False

    if args.csv:
        # --- CSV-driven mode: read design parameters from CSV ---
        print(f"Reading design parameters from CSV: {args.csv}")
        df = pd.read_csv(args.csv)

        # Allow overwriting spatial and angular resolution
        if args.grid_cell_size is not None:
            print(f"Overriding grid_cell_size with {args.grid_cell_size}")
            df["grid_cell_size"] = args.grid_cell_size
        if args.quad_order is not None:
            print(f"Overriding quad_order with {args.quad_order}")
            df["quad_order"] = args.quad_order

        design_params = df[
            ["abs_blue", "scatter_white", "grid_cell_size", "quad_order"]
        ].to_numpy()
        design_param_names = np.array(
            ["absorption_blue", "scattering_white", "grid_cl", "grid_quad_order"]
        )
    elif load_from_npz:  # TODO
        raise NotImplementedError
        design_params, design_param_names = load_lattice_samples_from_npz(
            "sampling/pilot-study-samples-hohlraum-05-29-24.npz"
        )
        exit("TODO")
    else:
        # --- Define parameter ranges ---

        #  characteristic length of the cells:  #grid cells = O(1/cell_size^2)
        parameter_range_grid_cell_size = [0.01]

        # quadrature order (must be an even number):  #velocity grid cells = O(order^2)
        parameter_range_quad_order = [4]

        # Prescribed range for LATTICE_DSGN_ABSORPTION_BLUE
        parameter_range_abs_blue = [10]  # default: 10 #10, 50, 100
        # Prescribed range for LATTICE_DSGN_SCATTER_WHITE
        parameter_range_scatter_white = [1]  # default: 1 0.1, 0.5, 1, 5, 10

        design_params, design_param_names = create_lattice_samples_from_param_range(
            parameter_range_grid_cell_size,
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
            use_cuda=use_cuda,
        )

        user = read_username_from_config("./slurm_config.txt")
        if user:
            print("Executing slurm scripts with user " + user)
            execute_slurm_scripts(directory, user)
            wait_for_slurm_jobs(user=user, sleep_interval=10)
        else:
            print("Username could not be read from config file.")

        qois = call_models(
            design_params, hpc_operation_count=2, use_cuda=use_cuda
        )
    else:
        qois = call_models(
            design_params,
            hpc_operation_count=0,
            rectangular_mesh=rectangular_mesh,
            singularity_hpc=singularity_hpc,
            use_cuda=use_cuda,
        )

    if args.csv:
        # Add all QOI columns to the CSV
        qoi_names = get_qois_col_names()
        for i in range(qois.shape[1]):
            df[qoi_names[i]] = qois[:, i]
        df.to_csv(args.csv, index=False)
        print(f"Updated CSV with QOI columns: {args.csv}")
    else:
        np.savez(
            "benchmarks/lattice/sn_study_lattice.npz",
            qois=qois,
            design_params=design_params,
            qoi_column_names=get_qois_col_names(),
            design_param_column_names=design_param_names,
        )

    print("design parameter matrix")
    print(design_param_names)
    print(design_params)
    print("quantities of interest:")
    print(get_qois_col_names())
    print(qois)

    print("======== Finished ===========")
    return 0


def call_models(
    design_params,
    hpc_operation_count,
    singularity_hpc=True,
    rectangular_mesh=False,
    use_cuda=False,
):
    qois = []
    for column in design_params:
        input = column.tolist()
        print(input)
        input.append(hpc_operation_count)
        input.append(singularity_hpc)
        input.append(rectangular_mesh)
        input.append(use_cuda)

        res = model([input])
        qois.append(res[0])

    return np.array(qois)


if __name__ == "__main__":
    main()
