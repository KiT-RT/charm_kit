import numpy as np
import pytest

from src.models import hohlraum as hohlraum_model
from src.models import lattice as lattice_model


@pytest.fixture
def lattice_mocks(monkeypatch):
    calls = {"raw": [], "container": [], "slurm": []}
    base_config = {"LOG_DIR": "result/logs", "OUTPUT_DIR": "result"}

    def update_parameter(values, key, new_value):
        updated = values.copy()
        updated[key] = new_value
        return updated

    monkeypatch.setattr(lattice_model, "read_config_file", lambda _: base_config.copy())
    monkeypatch.setattr(lattice_model, "update_lattice_mesh_file", lambda *_, **__: "mesh.su2")
    monkeypatch.setattr(lattice_model, "update_parameter", update_parameter)
    monkeypatch.setattr(lattice_model, "write_config_file", lambda **_: None)
    monkeypatch.setattr(lattice_model, "remove_files", lambda *_: None)
    monkeypatch.setattr(
        lattice_model,
        "write_slurm_file",
        lambda *args, **kwargs: calls["slurm"].append((args, kwargs)),
    )
    monkeypatch.setattr(lattice_model, "generate_log_filename", lambda _: "result/logs/lattice_run")
    monkeypatch.setattr(
        lattice_model,
        "read_csv_file",
        lambda _: {
            "Wall_time_[s]": 1.0,
            "Mass": 2.0,
            "Cur_absorption": 3.0,
            "Total_absorption": 4.0,
            "Cur_outflow_P1": 5.0,
            "Total_outflow_P1": 6.0,
            "Cur_outflow_P2": 7.0,
            "Total_outflow_P2": 8.0,
        },
    )
    monkeypatch.setattr(
        lattice_model,
        "run_cpp_simulation",
        lambda config, quiet=False: calls["raw"].append((config, quiet)),
    )
    monkeypatch.setattr(
        lattice_model,
        "run_cpp_simulation_containerized",
        lambda config, use_cuda=False, quiet=False: calls["container"].append(
            (config, use_cuda, quiet)
        ),
    )
    monkeypatch.setattr(
        lattice_model.os.path, "exists", lambda path: str(path).endswith(".csv")
    )
    monkeypatch.setattr(lattice_model.os, "remove", lambda *_: None)
    return calls


@pytest.fixture
def hohlraum_mocks(monkeypatch):
    calls = {"raw": [], "container": [], "slurm": []}
    base_config = {"LOG_DIR": "result/logs", "OUTPUT_DIR": "result", "TIME_FINAL": 1.0}

    def update_parameter(values, key, new_value):
        updated = values.copy()
        updated[key] = new_value
        return updated

    def integrated_probes(*_, **__):
        result = {}
        for probe in range(4):
            for moment in range(3):
                result[f"Probe {probe} u_{moment}"] = [float(i) for i in range(10)]
        return result

    monkeypatch.setattr(hohlraum_model, "read_config_file", lambda _: base_config.copy())
    monkeypatch.setattr(
        hohlraum_model, "update_var_hohlraum_mesh_file", lambda **_: "mesh.su2"
    )
    monkeypatch.setattr(hohlraum_model, "update_parameter", update_parameter)
    monkeypatch.setattr(hohlraum_model, "write_config_file", lambda **_: None)
    monkeypatch.setattr(hohlraum_model, "remove_files", lambda *_: None)
    monkeypatch.setattr(
        hohlraum_model,
        "write_slurm_file",
        lambda *args, **kwargs: calls["slurm"].append((args, kwargs)),
    )
    monkeypatch.setattr(hohlraum_model, "generate_log_filename", lambda _: "result/logs/hohlraum_run")
    monkeypatch.setattr(
        hohlraum_model,
        "read_csv_file",
        lambda _: {
            "Wall_time_[s]": 1.0,
            "Mass": 2.0,
            "Cumulated_absorption_center": 3.0,
            "Cumulated_absorption_vertical_wall": 4.0,
            "Cumulated_absorption_horizontal_wall": 5.0,
        },
    )
    monkeypatch.setattr(
        hohlraum_model,
        "get_integrated_hohlraum_probe_moments",
        integrated_probes,
    )
    monkeypatch.setattr(
        hohlraum_model,
        "run_cpp_simulation",
        lambda config, quiet=False: calls["raw"].append((config, quiet)),
    )
    monkeypatch.setattr(
        hohlraum_model,
        "run_cpp_simulation_containerized",
        lambda config, use_cuda=False, quiet=False: calls["container"].append(
            (config, use_cuda, quiet)
        ),
    )
    monkeypatch.setattr(
        hohlraum_model.os.path, "exists", lambda path: str(path).endswith(".csv")
    )
    monkeypatch.setattr(hohlraum_model.os, "remove", lambda *_: None)
    return calls


def test_lattice_model_local_raw_operation(lattice_mocks):
    params = [[10.0, 1.0, 0.01, 4, 0, 0, False, False, True]]
    qois = lattice_model.model(params)[0]
    assert len(qois) == 8
    assert len(lattice_mocks["raw"]) == 1
    assert lattice_mocks["raw"][0][1] is True
    assert lattice_mocks["container"] == []


def test_lattice_model_local_cuda_uses_container(lattice_mocks):
    params = [[10.0, 1.0, 0.01, 4, 0, 0, False, True, False]]
    qois = lattice_model.model(params)[0]
    assert len(qois) == 8
    assert len(lattice_mocks["container"]) == 1
    assert lattice_mocks["container"][0][1] is True
    assert lattice_mocks["raw"] == []


def test_lattice_model_slurm_operation_writes_script(lattice_mocks):
    params = [[10.0, 1.0, 0.01, 4, 1, 1, False, False, False]]
    qois = lattice_model.model(params)[0]
    assert qois == [0] * 8
    assert len(lattice_mocks["slurm"]) == 1
    assert lattice_mocks["raw"] == []
    assert lattice_mocks["container"] == []


def test_lattice_model_postprocess_operation_reads_qois(lattice_mocks):
    params = [[10.0, 1.0, 0.01, 4, 2, 1, False, False, False]]
    qois = lattice_model.model(params)[0]
    assert len(qois) == 8
    assert qois[0] == 1.0
    assert lattice_mocks["raw"] == []
    assert lattice_mocks["container"] == []


def test_lattice_model_rejects_cuda_with_slurm():
    params = [[10.0, 1.0, 0.01, 4, 1, 1, False, True, False]]
    with pytest.raises(ValueError, match="CUDA mode with SLURM is not supported"):
        lattice_model.model(params)


def test_hohlraum_model_local_raw_operation(hohlraum_mocks):
    params = [[0.4, -0.4, 0.4, -0.4, -0.6, 0.6, 0.0, 0.0, 0.01, 6, 0, 0, False, True]]
    qois = hohlraum_model.model(params)[0]
    assert len(qois) == 125
    assert len(hohlraum_mocks["raw"]) == 1
    assert hohlraum_mocks["raw"][0][1] is True
    assert hohlraum_mocks["container"] == []


def test_hohlraum_model_local_cuda_uses_container(hohlraum_mocks):
    params = [[0.4, -0.4, 0.4, -0.4, -0.6, 0.6, 0.0, 0.0, 0.01, 6, 0, 0, True, False]]
    qois = hohlraum_model.model(params)[0]
    assert len(qois) == 125
    assert len(hohlraum_mocks["container"]) == 1
    assert hohlraum_mocks["container"][0][1] is True
    assert hohlraum_mocks["raw"] == []


def test_hohlraum_model_slurm_operation_writes_script(hohlraum_mocks):
    params = [[0.4, -0.4, 0.4, -0.4, -0.6, 0.6, 0.0, 0.0, 0.01, 6, 1, 1, False, False]]
    qois = hohlraum_model.model(params)[0]
    assert qois == [0] * 125
    assert len(hohlraum_mocks["slurm"]) == 1
    assert hohlraum_mocks["raw"] == []
    assert hohlraum_mocks["container"] == []


def test_hohlraum_model_postprocess_operation_reads_qois(hohlraum_mocks):
    params = [[0.4, -0.4, 0.4, -0.4, -0.6, 0.6, 0.0, 0.0, 0.01, 6, 2, 1, False, False]]
    qois = hohlraum_model.model(params)[0]
    assert len(qois) == 125
    assert qois[0] == 1.0
    assert hohlraum_mocks["raw"] == []
    assert hohlraum_mocks["container"] == []


def test_hohlraum_model_rejects_cuda_with_slurm():
    params = [[0.4, -0.4, 0.4, -0.4, -0.6, 0.6, 0.0, 0.0, 0.01, 6, 1, 1, True, False]]
    with pytest.raises(ValueError, match="CUDA mode with SLURM is not supported"):
        hohlraum_model.model(params)
