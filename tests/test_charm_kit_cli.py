from src import cli


def test_submit_injects_slurm_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_dispatch", lambda case, args: calls.append((case, args)) or 0)

    assert cli.main(["submit", "lattice", "--cuda"]) == 0

    assert calls == [("lattice", ["--slurm", "--cuda"])]


def test_run_preserves_runner_args(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_dispatch", lambda case, args: calls.append((case, args)) or 0)

    assert cli.main(["run", "hohlraum", "--config", "cfg.toml"]) == 0

    assert calls == [("hohlraum", ["--config", "cfg.toml"])]
