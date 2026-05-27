
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![GitHub Stars](https://img.shields.io/github/stars/KiT-RT/CharmKiT)](https://github.com/KiT-RT/CharmKiT/stargazers)
[![Tests](https://github.com/KiT-RT/CharmKiT/actions/workflows/tests.yml/badge.svg)](https://github.com/KiT-RT/CharmKiT/actions/workflows/tests.yml)

# CharmKiT: A wrapper for KiT-RT to conduct simulation sweeps fast


CharmKiT is a benchmarking suite for the CharmNet project, providing automated parameter studies and test case management for the [KiT-RT PDE simulator](https://kit-rt.readthedocs.io/en/develop/index.html). It enables reproducible runs of radiative transfer test cases such as the lattice and hohlraum setups, using Python scripts to manage parameter sweeps, configuration, and result collection. CharmKiT supports both high-performance computing (HPC) and local (no-HPC) execution modes, leveraging Apptainer/Singularity containers for reproducibility.



## Start here

CharmKiT is the Python workflow layer for KiT-RT. Use `kitrt_code` directly when you want to build or modify the C++ solver; use CharmKiT when you want reproducible lattice or hohlraum sweeps, CSV-driven runs, SLURM submission, or QOI collection.

| User goal | Use this | First command |
|---|---|---|
| Run one deterministic solver config | `kitrt_code` | `./build_omp/KiT-RT ...` |
| Run a local lattice sweep | CharmKiT | `poetry run charm-kit run lattice --singularity` |
| Run a local hohlraum sweep | CharmKiT | `poetry run charm-kit run hohlraum --singularity` |
| Submit a CPU SLURM sweep | CharmKiT | `poetry run charm-kit submit lattice --singularity` |
| Submit a CUDA SLURM sweep | CharmKiT | `poetry run charm-kit submit lattice --cuda` |

## Installation

Preliminaries:

1. Install [Apptainer](https://apptainer.org/) or [Singularity](https://docs.sylabs.io/guides/latest/user-guide/quick_start.html). Set `KITRT_CONTAINER_RUNTIME=apptainer` or `KITRT_CONTAINER_RUNTIME=singularity` if the runtime is not on `PATH` or you need to force one.

2. Clone the CharmKiT repository:

   ```bash
   git clone git@github.com:KiT-RT/CharmKiT.git
   cd CharmKiT
   ```

3. Install Poetry and create the project environment:

   ```bash
   python3 -m pip install --user poetry
   poetry install
   ```

4. Install [KiT-RT](https://github.com/KiT-RT/kitrt_code) into `./kitrt_code/` using the provided installer:

   ```bash
   bash install_kitrt.sh
   ```

   The installer builds or reuses the CPU container and binary. If a CUDA GPU is detected, it also prepares the CUDA container and `build_singularity_cuda` binary. To avoid root builds on an HPC system, provide existing SIF files in `kitrt_code/tools/singularity/`, set `KITRT_CONTAINER_BUILD=skip`, or set `KITRT_CPU_IMAGE_URI` / `KITRT_CUDA_IMAGE_URI` to an Apptainer/Singularity-compatible pull URI.

   ```bash
   KITRT_CONTAINER_RUNTIME=apptainer KITRT_CONTAINER_BUILD=skip bash install_kitrt.sh
   ```

5. Update an existing KiT-RT checkout and rebuild inside existing containers:

   ```bash
   bash update_kitrt.sh
   ```

## Testing

Run unit tests from the repo root:

```bash
poetry install --with dev
poetry run pytest -q
```


## How CharmKiT Works

CharmKiT automates the setup, execution, and result collection for radiative transfer test cases using the KiT-RT solver. The workflow is managed by Python scripts (e.g., `run_hohlraum.py`, `run_lattice.py`) that:

- Define parameter sweeps for each test case (e.g., mesh size, quadrature order, absorption/scattering coefficients).
- Generate the necessary configuration files for KiT-RT.
- Run the KiT-RT solver inside a Singularity container for each parameter combination.
- Collect and save the results (e.g., quantities of interest) as `.npz` files for further analysis.

Scripts support both HPC (SLURM) and local (no-HPC) execution. 


## Running CharmKiT

CharmKiT provides a `charm-kit` CLI plus backward-compatible root scripts. Use Poetry to run commands in the project environment.

```bash
poetry run charm-kit run lattice --help
poetry run charm-kit run hohlraum --help
poetry run python run_lattice.py --help
poetry run python run_hohlraum.py --help
```

`charm-kit run <case>` runs locally by default. `charm-kit submit <case>` adds `--slurm` and generates/submits SLURM scripts. The lattice and hohlraum runners share execution flags, but design-parameter flags are test-case specific.

Execution and I/O flags:

- `--slurm`: Submit jobs through SLURM.
- `--singularity`: Run KiT-RT through the CPU container image.
- `--cuda`: Run KiT-RT through the CUDA Singularity image (`--nv` is added automatically).
- `--csv CSV`: Read design parameters from CSV and write QOIs back to that CSV.
- `--config CONFIG`: Path to a TOML hyperparameter file.
- `-q`, `--quiet`: Suppress solver stdout/stderr output.

Shared parameter override flags:

- `--grid-cell-size GRID_CELL_SIZE`: Override spatial resolution for all runs.
- `--quad-order QUAD_ORDER`: Override angular quadrature order for all runs (must be even).

Lattice-specific parameter flags:

- `--abs-blue ABS_BLUE [ABS_BLUE ...]`: Override absorption coefficient(s) in blue lattice cells.
- `--scatter-white SCATTER_WHITE [SCATTER_WHITE ...]`: Override scattering coefficient(s) in white lattice cells.

Hohlraum-specific parameter flags:

- `--green-center-x GREEN_CENTER_X [GREEN_CENTER_X ...]`
- `--green-center-y GREEN_CENTER_Y [GREEN_CENTER_Y ...]`
- `--red-right-top RED_RIGHT_TOP [RED_RIGHT_TOP ...]`
- `--red-right-bottom RED_RIGHT_BOTTOM [RED_RIGHT_BOTTOM ...]`
- `--red-left-top RED_LEFT_TOP [RED_LEFT_TOP ...]`
- `--red-left-bottom RED_LEFT_BOTTOM [RED_LEFT_BOTTOM ...]`
- `--horizontal-left HORIZONTAL_LEFT [HORIZONTAL_LEFT ...]`
- `--horizontal-right HORIZONTAL_RIGHT [HORIZONTAL_RIGHT ...]`

Default hyperparameter files:
- `benchmarks/lattice/hyperparams.toml`
- `benchmarks/hohlraum/hyperparams.toml`

Precedence for hyperparameters is:
1. Command-line arguments
2. TOML values
3. Script hardcoded defaults

### Supported run setups

1. **Local mode, raw (no Singularity)**

   ```bash
   poetry run python run_lattice.py
   # or
   poetry run python run_hohlraum.py
   ```

   Uses local executable: `./kitrt_code/build/KiT-RT`.

2. **Local mode + Singularity (CPU)**

   ```bash
   poetry run python run_lattice.py --singularity
   # or
   poetry run python run_hohlraum.py --singularity
   ```

   Uses image/executable:
   `kitrt_code/tools/singularity/kit_rt.sif` and `./kitrt_code/build_singularity/KiT-RT`. Set `KITRT_CONTAINER_RUNTIME=apptainer` to run with Apptainer instead of Singularity.

3. **Local mode + Singularity + GPU**

   ```bash
   poetry run python run_lattice.py --cuda
   # or
   poetry run python run_hohlraum.py --cuda
   ```

   Uses image/executable:
   `kitrt_code/tools/singularity/kit_rt_MPI_cuda.sif` and `./kitrt_code/build_singularity_cuda/KiT-RT`.
   CUDA runs are dispatched as:
   `$KITRT_CONTAINER_RUNTIME exec --nv ... mpirun -np <gpu_count> ./kitrt_code/build_singularity_cuda/KiT-RT ...`.
   `<gpu_count>` is auto-detected from `CUDA_VISIBLE_DEVICES` or `nvidia-smi`.
   Override rank count with `KITRT_CUDA_MPI_RANKS=<N>`.

4. **SLURM mode, raw (no Singularity)**

   ```bash
   poetry run python run_lattice.py --slurm
   # or
   poetry run python run_hohlraum.py --slurm
   ```

   Generated SLURM scripts call: `srun ./kitrt_code/build/KiT-RT ...`.

5. **SLURM mode + Singularity/Apptainer (CPU)**

   ```bash
   poetry run python run_lattice.py --slurm --singularity
   # or
   poetry run python run_hohlraum.py --slurm --singularity
   ```

   Generated SLURM scripts call:
   `srun "$KITRT_CONTAINER_RUNTIME" exec kitrt_code/tools/singularity/kit_rt.sif ./kitrt_code/build_singularity/KiT-RT ...`.

6. **SLURM mode + Singularity/Apptainer + GPU**

   ```bash
   poetry run charm-kit submit lattice --cuda
   # or
   poetry run charm-kit submit hohlraum --cuda
   ```

   Generated SLURM scripts use `--nv`, the CUDA SIF at `kitrt_code/tools/singularity/kit_rt_MPI_cuda.sif`, and `mpirun -np "$KITRT_CUDA_MPI_RANKS"`. If `KITRT_CUDA_MPI_RANKS` is unset, the script falls back to `SLURM_GPUS_ON_NODE` and then `1`. Update `slurm_template.sh` with the account, partition, and GPU directives required by your site.


# Test Case Descriptions

## 1. Lattice Test Case

The lattice test case models an isotropic radiative source at the center of a 2D domain, surrounded by a periodic arrangement of blue, red, and white squares. Each color represents a different material with specific absorption, scattering, and source properties:
![Lattice test case](documentation/lattice_setup.png)

| Region | Absorption | Scattering | Source |
|--------|------------|------------|--------|
| Blue   | 10         | 0          | 0      |
| Red    | 0          | 1          | 1      |
| White  | 0          | 1          | 0      |

The main design parameters are:
- Number of grid points per square side
- Quadrature order (velocity space)
- Absorption in blue squares
- Scattering in white squares

The script `run_lattice.py` automates parameter sweeps over these variables, generating KiT-RT config files and collecting results. Quantities of interest include mass, outflow and absorption metrics, as well as wall time. The mesh and configuration can be customized via script arguments.

See `benchmarks/lattice/` for config templates and mesh files.

## 2. Hohlraum Test Case

The hohlraum test case models linear radiative transfer in a symmetric 2D cavity with mixed inflow and void boundary segments. The geometry affects transport and absorption.
![Hohlraum test case](documentation/hohlraum.png)

The main design parameters are:
- Mesh characteristic length (spatial resolution)
- Quadrature order (velocity space)
- Capsule center location (`x`, `y`)
- Left and right absorber geometry parameters (top, bottom, and horizontal wall positions)

The script `run_hohlraum.py` automates parameter sweeps over these variables, generating KiT-RT config files and collecting results. Quantities of interest include mass, wall time, cumulative absorption metrics in key regions, and probe-moment summaries. The mesh and configuration can be customized via script arguments.

See `benchmarks/hohlraum/` for config templates and mesh files.

---


For more details on the scientific background and test case motivation, see the accompanying paper and the documentation in `documentation/`.

---

## Citation

If you use charm_kit or the provided benchmarks in your research, please cite:

```bibtex
@misc{schotthöfer2025referencesolutionslinearradiation,
      title={Reference solutions for linear radiation transport: the Hohlraum and Lattice benchmarks}, 
      author={Steffen Schotthöfer and Cory Hauck},
      year={2025},
      eprint={2505.17284},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph},
      url={https://arxiv.org/abs/2505.17284}, 
}
```
