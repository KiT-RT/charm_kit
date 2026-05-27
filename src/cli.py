import sys


USAGE = """usage:
  charm-kit run lattice [runner options]
  charm-kit run hohlraum [runner options]
  charm-kit submit lattice [runner options]
  charm-kit submit hohlraum [runner options]

Use "charm-kit run lattice --help" or "charm-kit run hohlraum --help" for
case-specific options.
"""


def _dispatch(case_name, runner_args):
    if case_name == "lattice":
        import run_lattice

        sys.argv = ["run_lattice.py", *runner_args]
        return run_lattice.main()
    if case_name == "hohlraum":
        import run_hohlraum

        sys.argv = ["run_hohlraum.py", *runner_args]
        return run_hohlraum.main()
    raise SystemExit(f"ERROR: unknown case {case_name!r}\n\n{USAGE}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    command = argv.pop(0)
    if command not in {"run", "submit"}:
        raise SystemExit(f"ERROR: unknown command {command!r}\n\n{USAGE}")
    if not argv:
        raise SystemExit(f"ERROR: missing benchmark case\n\n{USAGE}")

    case_name = argv.pop(0)
    runner_args = argv
    if (
        command == "submit"
        and "--slurm" not in runner_args
        and "--use-slurm" not in runner_args
    ):
        runner_args = ["--slurm", *runner_args]

    return _dispatch(case_name, runner_args)


if __name__ == "__main__":
    raise SystemExit(main())
