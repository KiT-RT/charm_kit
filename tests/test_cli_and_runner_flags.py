import argparse
import sys

import numpy as np
import pandas as pd
import pytest

import run_hohlraum
import run_lattice
from src import general_utils


def lattice_args(**overrides):
    values = {
        "use_slurm": False,
        "use_singularity": False,
        "cuda": False,
        "quiet": False,
        "csv": None,
        "config": None,
        "grid_cell_size": None,
        "quad_order": None,
        "abs_blue": None,
        "scatter_white": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def hohlraum_args(**overrides):
    values = {
        "use_slurm": False,
        "use_singularity": False,
        "cuda": False,
        "quiet": False,
        "csv": None,
        "config": None,
        "grid_cell_size": None,
        "quad_order": None,
        "green_center_x": None,
        "green_center_y": None,
        "red_right_top": None,
        "red_right_bottom": None,
        "red_left_top": None,
        "red_left_bottom": None,
        "horizontal_left": None,
        "horizontal_right": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_parse_lattice_args_supports_current_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_lattice.py",
            "--slurm",
            "--singularity",
            "--cuda",
            "--quiet",
            "--csv",
            "in.csv",
            "--grid-cell-size",
            "0.02",
            "--quad-order",
            "6",
            "--config",
            "cfg.toml",
            "--abs-blue",
            "10.0",
            "11.0",
            "--scatter-white",
            "1.0",
            "2.0",
        ],
    )
    args = general_utils.parse_lattice_args()
    assert args.use_slurm is True
    assert args.use_singularity is True
    assert args.cuda is True
    assert args.quiet is True
    assert args.csv == "in.csv"
    assert args.grid_cell_size == 0.02
    assert args.quad_order == 6
    assert args.config == "cfg.toml"
    assert args.abs_blue == [10.0, 11.0]
    assert args.scatter_white == [1.0, 2.0]
    assert not hasattr(args, "load_from_npz")


def test_parse_hohlraum_args_supports_current_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_hohlraum.py",
            "--slurm",
            "--singularity",
            "--cuda",
            "--quiet",
            "--csv",
            "in.csv",
            "--grid-cell-size",
            "0.01",
            "--quad-order",
            "8",
            "--config",
            "cfg.toml",
            "--green-center-x",
            "-0.1",
            "0.1",
            "--green-center-y",
            "-0.2",
            "0.2",
            "--red-right-top",
            "0.5",
            "--red-right-bottom",
            "-0.5",
            "--red-left-top",
            "0.5",
            "--red-left-bottom",
            "-0.5",
            "--horizontal-left",
            "-0.6",
            "--horizontal-right",
            "0.6",
        ],
    )
    args = general_utils.parse_hohlraum_args()
    assert args.use_slurm is True
    assert args.use_singularity is True
    assert args.cuda is True
    assert args.quiet is True
    assert args.csv == "in.csv"
    assert args.grid_cell_size == 0.01
    assert args.quad_order == 8
    assert args.config == "cfg.toml"
    assert args.green_center_x == [-0.1, 0.1]
    assert args.green_center_y == [-0.2, 0.2]
    assert args.red_right_top == [0.5]
    assert args.red_right_bottom == [-0.5]
    assert args.red_left_top == [0.5]
    assert args.red_left_bottom == [-0.5]
    assert args.horizontal_left == [-0.6]
    assert args.horizontal_right == [0.6]
    assert not hasattr(args, "load_from_npz")


def test_lattice_main_rejects_cuda_with_slurm(monkeypatch):
    monkeypatch.setattr(
        run_lattice, "parse_lattice_args", lambda: lattice_args(use_slurm=True, cuda=True)
    )
    with pytest.raises(SystemExit, match="--cuda cannot be combined with --slurm"):
        run_lattice.main()


def test_hohlraum_main_rejects_cuda_with_slurm(monkeypatch):
    monkeypatch.setattr(
        run_hohlraum,
        "parse_hohlraum_args",
        lambda: hohlraum_args(use_slurm=True, cuda=True),
    )
    with pytest.raises(SystemExit, match="--cuda cannot be combined with --slurm"):
        run_hohlraum.main()


def test_lattice_main_local_cuda_forces_containerized_mode(monkeypatch):
    monkeypatch.setattr(
        run_lattice, "parse_lattice_args", lambda: lattice_args(cuda=True, use_singularity=False)
    )
    monkeypatch.setattr(run_lattice, "load_toml_hyperparameters", lambda _: {})
    monkeypatch.setattr(
        run_lattice,
        "create_lattice_samples_from_param_range",
        lambda *_: (
            np.array([[10.0, 1.0, 0.01, 4]]),
            np.array(["absorption_blue", "scattering_white", "grid_cl", "grid_quad_order"]),
        ),
    )
    calls = []

    def fake_call_models(*args, **kwargs):
        calls.append(kwargs)
        return np.array([[1.0] * 8])

    monkeypatch.setattr(run_lattice, "call_models", fake_call_models)
    monkeypatch.setattr(run_lattice.np, "savez", lambda *_, **__: None)

    assert run_lattice.main() == 0
    assert len(calls) == 1
    assert calls[0]["hpc_operation_count"] == 0
    assert calls[0]["singularity_hpc"] is True
    assert calls[0]["use_cuda"] is True


def test_hohlraum_main_local_cuda_forces_containerized_mode(monkeypatch):
    monkeypatch.setattr(
        run_hohlraum,
        "parse_hohlraum_args",
        lambda: hohlraum_args(cuda=True, use_singularity=False),
    )
    monkeypatch.setattr(run_hohlraum, "load_toml_hyperparameters", lambda _: {})
    monkeypatch.setattr(
        run_hohlraum,
        "create_hohlraum_samples_from_param_range",
        lambda *_: (np.zeros((10, 1)), np.array(["p"] * 10)),
    )
    calls = []

    def fake_call_models(*args, **kwargs):
        calls.append(kwargs)
        return np.array([[1.0] * len(run_hohlraum.get_qois_col_names())])

    monkeypatch.setattr(run_hohlraum, "call_models", fake_call_models)
    monkeypatch.setattr(run_hohlraum.np, "savez", lambda *_, **__: None)

    assert run_hohlraum.main() == 0
    assert len(calls) == 1
    assert calls[0]["hpc_operation_count"] == 0
    assert calls[0]["singularity_hpc"] is True
    assert calls[0]["use_cuda"] is True


def test_lattice_main_slurm_runs_stage_1_and_2(monkeypatch):
    monkeypatch.setattr(
        run_lattice,
        "parse_lattice_args",
        lambda: lattice_args(use_slurm=True, use_singularity=True),
    )
    monkeypatch.setattr(run_lattice, "load_toml_hyperparameters", lambda _: {})
    monkeypatch.setattr(
        run_lattice,
        "create_lattice_samples_from_param_range",
        lambda *_: (np.array([[10.0, 1.0, 0.01, 4]]), np.array(["a", "b", "c", "d"])),
    )

    calls = []

    def fake_call_models(*args, **kwargs):
        calls.append(kwargs["hpc_operation_count"])
        return np.array([[1.0] * 8])

    monkeypatch.setattr(run_lattice, "call_models", fake_call_models)
    monkeypatch.setattr(run_lattice, "delete_slurm_scripts", lambda *_: None)
    monkeypatch.setattr(run_lattice, "read_username_from_config", lambda *_: "alice")
    monkeypatch.setattr(run_lattice, "execute_slurm_scripts", lambda *_: None)
    monkeypatch.setattr(run_lattice, "wait_for_slurm_jobs", lambda **_: None)
    monkeypatch.setattr(run_lattice.np, "savez", lambda *_, **__: None)

    assert run_lattice.main() == 0
    assert calls == [1, 2]


def test_hohlraum_main_slurm_runs_stage_1_and_2(monkeypatch):
    monkeypatch.setattr(
        run_hohlraum,
        "parse_hohlraum_args",
        lambda: hohlraum_args(use_slurm=True, use_singularity=True),
    )
    monkeypatch.setattr(run_hohlraum, "load_toml_hyperparameters", lambda _: {})
    monkeypatch.setattr(
        run_hohlraum,
        "create_hohlraum_samples_from_param_range",
        lambda *_: (np.zeros((10, 1)), np.array(["p"] * 10)),
    )

    calls = []

    def fake_call_models(*args, **kwargs):
        calls.append(kwargs["hpc_operation_count"])
        return np.array([[1.0] * len(run_hohlraum.get_qois_col_names())])

    monkeypatch.setattr(run_hohlraum, "call_models", fake_call_models)
    monkeypatch.setattr(run_hohlraum, "delete_slurm_scripts", lambda *_: None)
    monkeypatch.setattr(run_hohlraum, "read_username_from_config", lambda *_: "alice")
    monkeypatch.setattr(run_hohlraum, "execute_slurm_scripts", lambda *_: None)
    monkeypatch.setattr(run_hohlraum, "wait_for_slurm_jobs", lambda **_: None)
    monkeypatch.setattr(run_hohlraum.np, "savez", lambda *_, **__: None)

    assert run_hohlraum.main() == 0
    assert calls == [1, 2]


def test_lattice_main_csv_mode_updates_columns_and_qois(monkeypatch, tmp_path):
    csv_path = tmp_path / "lattice.csv"
    pd.DataFrame(
        [
            {
                "abs_blue": 10.0,
                "scatter_white": 1.0,
                "grid_cell_size": 0.01,
                "quad_order": 4,
            }
        ]
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr(
        run_lattice,
        "parse_lattice_args",
        lambda: lattice_args(
            csv=str(csv_path),
            abs_blue=[12.0],
            scatter_white=[2.0],
            grid_cell_size=0.02,
            quad_order=6,
        ),
    )
    monkeypatch.setattr(run_lattice, "load_toml_hyperparameters", lambda _: {})
    monkeypatch.setattr(run_lattice, "call_models", lambda *_, **__: np.array([[1.0] * 8]))
    monkeypatch.setattr(
        run_lattice.np,
        "savez",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("np.savez should not run in --csv mode")),
    )

    assert run_lattice.main() == 0
    output = pd.read_csv(csv_path)
    assert output.loc[0, "abs_blue"] == 12.0
    assert output.loc[0, "scatter_white"] == 2.0
    assert output.loc[0, "grid_cell_size"] == 0.02
    assert output.loc[0, "quad_order"] == 6
    for qoi_name in run_lattice.get_qois_col_names():
        assert qoi_name in output.columns


def test_hohlraum_main_csv_mode_updates_columns_and_qois(monkeypatch, tmp_path):
    csv_path = tmp_path / "hohlraum.csv"
    pd.DataFrame(
        [
            {
                "red_left_top": 0.4,
                "red_left_bottom": -0.4,
                "red_right_top": 0.4,
                "red_right_bottom": -0.4,
                "horizontal_left": -0.6,
                "horizontal_right": 0.6,
                "green_center_x": 0.0,
                "green_center_y": 0.0,
                "grid_cell_size": 0.01,
                "quad_order": 6,
            }
        ]
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr(
        run_hohlraum,
        "parse_hohlraum_args",
        lambda: hohlraum_args(
            csv=str(csv_path),
            green_center_x=[0.1],
            horizontal_right=[0.55],
            grid_cell_size=0.02,
            quad_order=8,
        ),
    )
    monkeypatch.setattr(run_hohlraum, "load_toml_hyperparameters", lambda _: {})
    monkeypatch.setattr(
        run_hohlraum,
        "call_models",
        lambda *_, **__: np.array([[1.0] * len(run_hohlraum.get_qois_col_names())]),
    )
    monkeypatch.setattr(
        run_hohlraum.np,
        "savez",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("np.savez should not run in --csv mode")),
    )

    assert run_hohlraum.main() == 0
    output = pd.read_csv(csv_path)
    assert output.loc[0, "green_center_x"] == 0.1
    assert output.loc[0, "horizontal_right"] == 0.55
    assert output.loc[0, "grid_cell_size"] == 0.02
    assert output.loc[0, "quad_order"] == 8
    for qoi_name in run_hohlraum.get_qois_col_names():
        assert qoi_name in output.columns


def test_lattice_main_rejects_odd_quad_order(monkeypatch):
    monkeypatch.setattr(
        run_lattice, "parse_lattice_args", lambda: lattice_args(quad_order=3)
    )
    monkeypatch.setattr(run_lattice, "load_toml_hyperparameters", lambda _: {})
    with pytest.raises(SystemExit, match="--quad-order must be an even number"):
        run_lattice.main()


def test_hohlraum_main_rejects_odd_quad_order(monkeypatch):
    monkeypatch.setattr(
        run_hohlraum, "parse_hohlraum_args", lambda: hohlraum_args(quad_order=5)
    )
    monkeypatch.setattr(run_hohlraum, "load_toml_hyperparameters", lambda _: {})
    with pytest.raises(SystemExit, match="--quad-order must be an even number"):
        run_hohlraum.main()


def test_lattice_call_models_appends_operation_and_execution_flags(monkeypatch):
    captured = []

    def fake_model(payload):
        captured.append(payload[0])
        return [[42.0] * 8]

    monkeypatch.setattr(run_lattice, "model", fake_model)
    design_params = np.array([[10.0, 1.0, 0.01, 4]])
    qois = run_lattice.call_models(
        design_params,
        hpc_operation_count=2,
        singularity_hpc=True,
        rectangular_mesh=False,
        use_cuda=True,
        quiet=True,
    )
    assert qois.shape == (1, 8)
    assert captured[0][-5:] == [2, True, False, True, True]


def test_hohlraum_call_models_uses_transposed_design_matrix(monkeypatch):
    captured = []

    def fake_model(payload):
        captured.append(payload[0])
        return [[7.0] * len(run_hohlraum.get_qois_col_names())]

    monkeypatch.setattr(run_hohlraum, "model", fake_model)
    design_params = np.arange(10).reshape(10, 1)
    qois = run_hohlraum.call_models(
        design_params,
        hpc_operation_count=1,
        singularity_hpc=False,
        use_cuda=False,
        quiet=True,
    )
    assert qois.shape == (1, len(run_hohlraum.get_qois_col_names()))
    assert captured[0][-4:] == [1, False, False, True]
