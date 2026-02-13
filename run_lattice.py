# import umbridge
import numpy as np
import pandas as pd

from src.config_utils import read_username_from_config
from src.models.lattice import get_qois_col_names, model

from src.simulation_utils import execute_slurm_scripts, wait_for_slurm_jobs
from src.general_utils import (
    create_lattice_samples_from_param_range,
    delete_slurm_scripts,
    load_toml_hyperparameters,
)

from src.general_utils import parse_lattice_args


def main():
    args = parse_lattice_args()
    print(f"HPC mode = { args.use_slurm}")
    print(f"HPC with singularity = { args.use_singularity}")
    print(f"CUDA mode = {args.cuda}")
    print(f"Quiet mode = {args.quiet}")

    hpc_operation = args.use_slurm  # Flag when using HPC cluster
    use_cuda = args.cuda
    if use_cuda and hpc_operation:
        raise SystemExit(
            "ERROR: --cuda cannot be combined with --slurm. "
            "GPU mode is supported only with Singularity in non-SLURM runs."
        )
    singularity_hpc = args.use_singularity or use_cuda
    if use_cuda and not args.use_singularity:
        print("CUDA mode requested; enabling Singularity execution.")

    # print(f"Use rectangular_mesh = {args.rectangular_mesh}")
    rectangular_mesh = False
    config_path = args.config or "benchmarks/lattice/hyperparams.toml"
    hyper = load_toml_hyperparameters(config_path)
    print(f"Lattice hyperparameter config = {config_path}")

    def as_list_or_none(value):
        if value is None:
            return None
        if isinstance(value, list):
            return value
        return [value]

    def override_csv_column(df, values, column_name, arg_name):
        if values is None:
            return
        values = as_list_or_none(values)
        print(f"Overriding {column_name} with {values}")
        if len(values) == 1:
            df[column_name] = values[0]
        elif len(values) == len(df):
            df[column_name] = values
        else:
            raise SystemExit(
                f"ERROR: In --csv mode, {arg_name} must provide either one value "
                "or exactly one value per CSV row."
            )

    if args.csv:
        # --- CSV-driven mode: read design parameters from CSV ---
        print(f"Reading design parameters from CSV: {args.csv}")
        df = pd.read_csv(args.csv)

        abs_blue_override = (
            args.abs_blue if args.abs_blue is not None else hyper.get("abs_blue")
        )
        scatter_white_override = (
            args.scatter_white
            if args.scatter_white is not None
            else hyper.get("scatter_white")
        )
        override_csv_column(df, abs_blue_override, "abs_blue", "--abs-blue")
        override_csv_column(
            df, scatter_white_override, "scatter_white", "--scatter-white"
        )
        # Allow overwriting spatial and angular resolution
        grid_cell_size_override = (
            args.grid_cell_size
            if args.grid_cell_size is not None
            else hyper.get("grid_cell_size")
        )
        quad_order_override = (
            args.quad_order if args.quad_order is not None else hyper.get("quad_order")
        )
        override_csv_column(
            df,
            grid_cell_size_override,
            "grid_cell_size",
            "--grid-cell-size",
        )
        override_csv_column(df, quad_order_override, "quad_order", "--quad-order")

        design_params = df[
            ["abs_blue", "scatter_white", "grid_cell_size", "quad_order"]
        ].to_numpy()
        design_param_names = np.array(
            ["absorption_blue", "scattering_white", "grid_cl", "grid_quad_order"]
        )
    else:
        # --- Define parameter ranges ---

        #  characteristic length of the cells:  #grid cells = O(1/cell_size^2)
        parameter_range_grid_cell_size = as_list_or_none(
            args.grid_cell_size if args.grid_cell_size is not None else hyper.get("grid_cell_size")
        ) or [0.01]

        # quadrature order (must be an even number):  #velocity grid cells = O(order^2)
        parameter_range_quad_order = as_list_or_none(
            args.quad_order if args.quad_order is not None else hyper.get("quad_order")
        ) or [4]
        if any(int(q) % 2 != 0 for q in parameter_range_quad_order):
            raise SystemExit("ERROR: --quad-order must be an even number.")

        # Prescribed range for LATTICE_DSGN_ABSORPTION_BLUE
        parameter_range_abs_blue = as_list_or_none(
            args.abs_blue if args.abs_blue is not None else hyper.get("abs_blue")
        ) or [10]
        # Prescribed range for LATTICE_DSGN_SCATTER_WHITE
        parameter_range_scatter_white = as_list_or_none(
            args.scatter_white
            if args.scatter_white is not None
            else hyper.get("scatter_white")
        ) or [1]

        design_params, design_param_names = create_lattice_samples_from_param_range(
            parameter_range_grid_cell_size,
            parameter_range_quad_order,
            parameter_range_abs_blue,
            parameter_range_scatter_white,
        )

    def safe_call_models(*model_args, **model_kwargs):
        try:
            return call_models(*model_args, **model_kwargs)
        except (RuntimeError, FileNotFoundError) as e:
            raise SystemExit(f"ERROR: {e}") from e

    if hpc_operation:
        print("==== Execute HPC version ====")
        directory = "./benchmarks/lattice/slurm_scripts/"

        delete_slurm_scripts(directory)  # delete existing slurm files for hohlraum
        safe_call_models(
            design_params,
            hpc_operation_count=1,
            singularity_hpc=singularity_hpc,
            rectangular_mesh=rectangular_mesh,
            use_cuda=use_cuda,
            quiet=args.quiet,
        )

        user = read_username_from_config("./slurm_config.txt")
        if user:
            print("Executing slurm scripts with user " + user)
            execute_slurm_scripts(directory, user)
            wait_for_slurm_jobs(user=user, sleep_interval=10)
        else:
            print("Username could not be read from config file.")

        qois = safe_call_models(
            design_params, hpc_operation_count=2, use_cuda=use_cuda, quiet=args.quiet
        )
    else:
        qois = safe_call_models(
            design_params,
            hpc_operation_count=0,
            rectangular_mesh=rectangular_mesh,
            singularity_hpc=singularity_hpc,
            use_cuda=use_cuda,
            quiet=args.quiet,
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
    quiet=False,
):
    qois = []
    for column in design_params:
        input = column.tolist()
        print(input)
        input.append(hpc_operation_count)
        input.append(singularity_hpc)
        input.append(rectangular_mesh)
        input.append(use_cuda)
        input.append(quiet)

        res = model([input])
        qois.append(res[0])

    return np.array(qois)


if __name__ == "__main__":
    main()
