import pytest
from src import simulation_utils


class _DummyResult:
    def __init__(self, returncode=0, stderr="", stdout=""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout


def test_run_cpp_simulation_containerized_cuda_uses_mpi(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return _DummyResult(returncode=0)

    monkeypatch.delenv("KITRT_CUDA_MPI_RANKS", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(simulation_utils.subprocess, "run", fake_run)

    simulation_utils.run_cpp_simulation_containerized(
        "tests/input/validation_tests/SN_solver_hpc/lattice_hpc_200_cuda_order2.cfg",
        use_cuda=True,
        quiet=True,
    )

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == [
        "singularity",
        "exec",
        "--nv",
        "kitrt_code/tools/singularity/kit_rt_MPI_cuda.sif",
        "mpirun",
        "-np",
        "2",
        "./kitrt_code/build_singularity_cuda/KiT-RT",
        "tests/input/validation_tests/SN_solver_hpc/lattice_hpc_200_cuda_order2.cfg",
    ]
    assert kwargs.get("stdout") == simulation_utils.subprocess.PIPE
    assert kwargs.get("stderr") == simulation_utils.subprocess.PIPE
    assert kwargs.get("text") is True


def test_run_cpp_simulation_containerized_cuda_respects_rank_override(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return _DummyResult(returncode=0)

    monkeypatch.setenv("KITRT_CUDA_MPI_RANKS", "4")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(simulation_utils.subprocess, "run", fake_run)

    simulation_utils.run_cpp_simulation_containerized(
        "benchmarks/lattice/example.cfg", use_cuda=True, quiet=True
    )

    assert len(calls) == 1
    assert calls[0][6] == "4"


def test_run_cpp_simulation_containerized_cuda_queries_nvidia_smi(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[:3] == [
            "nvidia-smi",
            "--query-gpu=index",
            "--format=csv,noheader",
        ]:
            return _DummyResult(returncode=0, stdout="0\n1\n2\n")
        return _DummyResult(returncode=0)

    monkeypatch.delenv("KITRT_CUDA_MPI_RANKS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(simulation_utils.subprocess, "run", fake_run)

    simulation_utils.run_cpp_simulation_containerized(
        "benchmarks/lattice/example.cfg", use_cuda=True, quiet=True
    )

    assert len(calls) == 2
    assert calls[0][0] == [
        "nvidia-smi",
        "--query-gpu=index",
        "--format=csv,noheader",
    ]
    assert calls[1][0][6] == "3"


def test_run_cpp_simulation_containerized_cuda_rejects_bad_rank_override(monkeypatch):
    monkeypatch.setenv("KITRT_CUDA_MPI_RANKS", "abc")
    with pytest.raises(RuntimeError, match="KITRT_CUDA_MPI_RANKS"):
        simulation_utils.run_cpp_simulation_containerized(
            "benchmarks/lattice/example.cfg", use_cuda=True, quiet=True
        )
