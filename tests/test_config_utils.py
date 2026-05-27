from src.config_utils import write_slurm_file


def test_write_slurm_file_cuda_uses_gpu_container(tmp_path, monkeypatch):
    template = tmp_path / "slurm_template.sh"
    template.write_text(
        "#SBATCH --mail-user=person@example.com\n"
        "### command below\n"
        "echo old command\n"
    )
    monkeypatch.chdir(tmp_path)

    write_slurm_file(
        "jobs",
        "case_a",
        "benchmarks/lattice/",
        singularity=True,
        use_cuda=True,
    )

    output = (tmp_path / "jobs" / "case_a.sh").read_text()
    assert "#SBATCH --mail-user=person@example.com" in output
    assert "echo old command" not in output
    assert "kit_rt_MPI_cuda.sif" in output
    assert "--nv" in output
    assert "CUDA_MPI_RANKS" in output
    assert "benchmarks/lattice/case_a.cfg" in output


def test_write_slurm_file_raw_uses_srun(tmp_path, monkeypatch):
    (tmp_path / "slurm_template.sh").write_text("#SBATCH -J test\n")
    monkeypatch.chdir(tmp_path)

    write_slurm_file("jobs", "case_b", "benchmarks/hohlraum/", singularity=False)

    output = (tmp_path / "jobs" / "case_b.sh").read_text()
    assert "srun ./kitrt_code/build/KiT-RT benchmarks/hohlraum/case_b.cfg" in output
