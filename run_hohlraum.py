import numpy as np
import pandas as pd
from src.models.hohlraum import get_qois_col_names, model


from src.config_utils import read_username_from_config
from src.simulation_utils import execute_slurm_scripts, wait_for_slurm_jobs
from src.general_utils import (
    create_hohlraum_samples_from_param_range,
    load_hohlraum_samples_from_csv,
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

        # Extract columns in the order expected by the hohlraum model:
        # left_red_top, left_red_bottom, right_red_top, right_red_bottom,
        # horizontal_left, horizontal_right, green_center_x, green_center_y,
        # grid_cell_size, quad_order
        design_params = df[
            [
                "red_left_top",
                "red_left_bottom",
                "red_right_top",
                "red_right_bottom",
                "horizontal_left",
                "horizontal_right",
                "green_center_x",
                "green_center_y",
                "grid_cell_size",
                "quad_order",
            ]
        ].to_numpy().T
        design_param_names = np.array(
            [
                "pos_red_left_top",
                "pos_red_left_bottom",
                "pos_red_right_top",
                "pos_red_right_bottom",
                "pos_red_left_horizontal",
                "pos_red_right_horizontal",
                "pos_green_x",
                "pos_green_y",
                "grid_cl",
                "grid_quad_order",
            ]
        )
    elif load_from_npz:
        design_params, design_param_names = load_hohlraum_samples_from_csv()
    else:
        # --- Define parameter ranges ---

        #  characteristic length of the cells:  #grid cells = O(1/cell_size^2)
        parameter_range_grid_cell_size = [0.0075]

        # quadrature order (must be an even number):  #velocity grid cells = O(order^2)
        parameter_range_quad_order = [6]
        # balance the two roughly (see Paper Fig 6)

        # Define the geometry settings of the test case

        parameter_range_green_center_x = [-0.1, 0.0, 0.1]  # Default: 0
        parameter_range_green_center_y = [-0.075, 0.0, 0.075]  # Default: 0
        parameter_range_red_right_top = [0.3, 0.4, 0.5]  # Default: 0.4
        parameter_range_red_right_bottom = [-0.5, -0.4, -0.3]  # Default: -0.4
        parameter_range_red_left_top = [0.3, 0.4, 0.5]  # Default: 0.4
        parameter_range_red_left_bottom = [-0.5, -0.4, -0.3]  # Default: -0.4
        parameter_range_horizontal_left = [-0.63, -0.6, -0.5]  # Default: -0.6
        parameter_range_horizontal_right = [0.5, 0.6, 0.63]  # Default: 0.6

        design_params, design_param_names = create_hohlraum_samples_from_param_range(
            parameter_range_grid_cell_size,
            parameter_range_quad_order,
            parameter_range_green_center_x,
            parameter_range_green_center_y,
            parameter_range_red_right_top,
            parameter_range_red_right_bottom,
            parameter_range_red_left_top,
            parameter_range_red_left_bottom,
            parameter_range_horizontal_left,
            parameter_range_horizontal_right,
        )

    if hpc_operation:
        print("==== Execute HPC version ====")
        directory = "./benchmarks/hohlraum/slurm_scripts/"
        user = read_username_from_config("./slurm_config.txt")

        delete_slurm_scripts(directory)  # delete existing slurm files for hohlraum
        call_models(
            design_params,
            hpc_operation_count=1,
            singularity_hpc=singularity_hpc,
            use_cuda=use_cuda,
        )
        # wait_for_slurm_jobs(user=user, sleep_interval=10)

        if user:
            print("Executing slurm scripts with user " + user)
            execute_slurm_scripts(directory, user)
            wait_for_slurm_jobs(user=user, sleep_interval=10)
        else:
            print("Username could not be read from slurm config file.")
        qois = call_models(
            design_params, hpc_operation_count=2, use_cuda=use_cuda
        )
    else:
        qois = call_models(
            design_params,
            hpc_operation_count=0,
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
            "benchmarks/hohlraum/sn_study_hohlraum.npz",
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


def call_models(design_params, hpc_operation_count, singularity_hpc=True, use_cuda=False):
    qois = []
    for column in design_params.T:
        input = column.tolist()
        input.append(hpc_operation_count)
        input.append(singularity_hpc)
        input.append(use_cuda)
        res = model([input])
        qois.append(res[0])

    return np.array(qois)


if __name__ == "__main__":
    main()
