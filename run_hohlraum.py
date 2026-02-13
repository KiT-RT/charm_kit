import numpy as np
import pandas as pd
from src.models.hohlraum import get_qois_col_names, model


from src.config_utils import read_username_from_config
from src.simulation_utils import execute_slurm_scripts, wait_for_slurm_jobs
from src.general_utils import (
    create_hohlraum_samples_from_param_range,
    delete_slurm_scripts,
    load_toml_hyperparameters,
)
from src.general_utils import parse_hohlraum_args


def main():
    args = parse_hohlraum_args()
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
    config_path = args.config or "benchmarks/hohlraum/hyperparams.toml"
    hyper = load_toml_hyperparameters(config_path)
    print(f"Hohlraum hyperparameter config = {config_path}")

    def as_list_or_none(value):
        if value is None:
            return None
        if isinstance(value, list):
            return value
        return [value]

    def override_csv_column(df, values, column_name, arg_name):
        values = as_list_or_none(values)
        if values is None:
            return
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

        override_csv_column(
            df,
            (
                args.green_center_x
                if args.green_center_x is not None
                else hyper.get("green_center_x")
            ),
            "green_center_x",
            "--green-center-x",
        )
        override_csv_column(
            df,
            (
                args.green_center_y
                if args.green_center_y is not None
                else hyper.get("green_center_y")
            ),
            "green_center_y",
            "--green-center-y",
        )
        override_csv_column(
            df,
            (
                args.red_right_top
                if args.red_right_top is not None
                else hyper.get("red_right_top")
            ),
            "red_right_top",
            "--red-right-top",
        )
        override_csv_column(
            df,
            (
                args.red_right_bottom
                if args.red_right_bottom is not None
                else hyper.get("red_right_bottom")
            ),
            "red_right_bottom",
            "--red-right-bottom",
        )
        override_csv_column(
            df,
            (
                args.red_left_top
                if args.red_left_top is not None
                else hyper.get("red_left_top")
            ),
            "red_left_top",
            "--red-left-top",
        )
        override_csv_column(
            df,
            (
                args.red_left_bottom
                if args.red_left_bottom is not None
                else hyper.get("red_left_bottom")
            ),
            "red_left_bottom",
            "--red-left-bottom",
        )
        override_csv_column(
            df,
            (
                args.horizontal_left
                if args.horizontal_left is not None
                else hyper.get("horizontal_left")
            ),
            "horizontal_left",
            "--horizontal-left",
        )
        override_csv_column(
            df,
            (
                args.horizontal_right
                if args.horizontal_right is not None
                else hyper.get("horizontal_right")
            ),
            "horizontal_right",
            "--horizontal-right",
        )
        # Allow overwriting spatial and angular resolution
        override_csv_column(
            df,
            (
                args.grid_cell_size
                if args.grid_cell_size is not None
                else hyper.get("grid_cell_size")
            ),
            "grid_cell_size",
            "--grid-cell-size",
        )
        override_csv_column(
            df,
            args.quad_order if args.quad_order is not None else hyper.get("quad_order"),
            "quad_order",
            "--quad-order",
        )

        # Extract columns in the order expected by the hohlraum model:
        # left_red_top, left_red_bottom, right_red_top, right_red_bottom,
        # horizontal_left, horizontal_right, green_center_x, green_center_y,
        # grid_cell_size, quad_order
        design_params = (
            df[
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
            ]
            .to_numpy()
            .T
        )
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
    else:
        # --- Define parameter ranges ---
        parameter_range_grid_cell_size = as_list_or_none(
            args.grid_cell_size
            if args.grid_cell_size is not None
            else hyper.get("grid_cell_size")
        ) or [0.0075]
        parameter_range_quad_order = as_list_or_none(
            args.quad_order if args.quad_order is not None else hyper.get("quad_order")
        ) or [6]
        if any(int(q) % 2 != 0 for q in parameter_range_quad_order):
            raise SystemExit("ERROR: --quad-order must be an even number.")

        parameter_range_green_center_x = as_list_or_none(
            args.green_center_x
            if args.green_center_x is not None
            else hyper.get("green_center_x")
        ) or [0.0]
        parameter_range_green_center_y = as_list_or_none(
            args.green_center_y
            if args.green_center_y is not None
            else hyper.get("green_center_y")
        ) or [0.0]
        parameter_range_red_right_top = as_list_or_none(
            args.red_right_top
            if args.red_right_top is not None
            else hyper.get("red_right_top")
        ) or [0.4]
        parameter_range_red_right_bottom = as_list_or_none(
            args.red_right_bottom
            if args.red_right_bottom is not None
            else hyper.get("red_right_bottom")
        ) or [-0.4]
        parameter_range_red_left_top = as_list_or_none(
            args.red_left_top
            if args.red_left_top is not None
            else hyper.get("red_left_top")
        ) or [0.4]
        parameter_range_red_left_bottom = as_list_or_none(
            args.red_left_bottom
            if args.red_left_bottom is not None
            else hyper.get("red_left_bottom")
        ) or [-0.4]
        parameter_range_horizontal_left = as_list_or_none(
            args.horizontal_left
            if args.horizontal_left is not None
            else hyper.get("horizontal_left")
        ) or [-0.6]
        parameter_range_horizontal_right = as_list_or_none(
            args.horizontal_right
            if args.horizontal_right is not None
            else hyper.get("horizontal_right")
        ) or [0.6]

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

    def safe_call_models(*model_args, **model_kwargs):
        try:
            return call_models(*model_args, **model_kwargs)
        except (RuntimeError, FileNotFoundError) as e:
            raise SystemExit(f"ERROR: {e}") from e

    if hpc_operation:
        print("==== Execute HPC version ====")
        directory = "./benchmarks/hohlraum/slurm_scripts/"
        user = read_username_from_config("./slurm_config.txt")

        delete_slurm_scripts(directory)  # delete existing slurm files for hohlraum
        safe_call_models(
            design_params,
            hpc_operation_count=1,
            singularity_hpc=singularity_hpc,
            use_cuda=use_cuda,
            quiet=args.quiet,
        )
        # wait_for_slurm_jobs(user=user, sleep_interval=10)

        if user:
            print("Executing slurm scripts with user " + user)
            execute_slurm_scripts(directory, user)
            wait_for_slurm_jobs(user=user, sleep_interval=10)
        else:
            print("Username could not be read from slurm config file.")
        qois = safe_call_models(
            design_params, hpc_operation_count=2, use_cuda=use_cuda, quiet=args.quiet
        )
    else:
        qois = safe_call_models(
            design_params,
            hpc_operation_count=0,
            singularity_hpc=singularity_hpc,
            use_cuda=use_cuda,
            quiet=args.quiet,
        )

    if args.csv:
        # Add all QOI columns to the CSV
        qoi_names = get_qois_col_names()
        qoi_df = pd.DataFrame(qois, columns=qoi_names, index=df.index)
        # Replace existing QOI columns in one concat to avoid DataFrame fragmentation.
        df = pd.concat([df.drop(columns=qoi_names, errors="ignore"), qoi_df], axis=1)
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


def call_models(
    design_params,
    hpc_operation_count,
    singularity_hpc=True,
    use_cuda=False,
    quiet=False,
):
    qois = []
    for column in design_params.T:
        input = column.tolist()
        input.append(hpc_operation_count)
        input.append(singularity_hpc)
        input.append(use_cuda)
        input.append(quiet)
        res = model([input])
        qois.append(res[0])

    return np.array(qois)


if __name__ == "__main__":
    main()
