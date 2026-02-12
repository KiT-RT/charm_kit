
# CharmKiT

CharmKiT is a benchmarking suite for the CharmNet project, providing automated parameter studies and test case management for the [KiT-RT PDE simulator](https://kit-rt.readthedocs.io/en/develop/index.html). It enables reproducible runs of radiative transfer test cases such as the lattice and hohlraum setups, using Python scripts to manage parameter sweeps, configuration, and result collection. CharmKiT supports both high-performance computing (HPC) and local (no-HPC) execution modes, leveraging Singularity containers for reproducibility.



## Installation

Preliminaries:

1. Install [Singularity](https://docs.sylabs.io/guides/latest/user-guide/quick_start.html) on your system.

2. Clone the `CharmKiT` Github repository:
   ```
   git clone git@github.com:ScSteffen/CharmKiT.git
   ```

3. Create a local Python environment and install requirements:
   ```
   python3 -m venv ./venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. Install [KiT-RT](https://github.com/CSMMLab/KiT-RT) as a submodule using the provided installer. (Requires root for container build.)
   ```
   sh install_KiT-RT.sh
   ```
   If updating KiT-RT:
   ```
   sh update_KiT-RT.sh
   ```
   If on a cluster without root, build the container locally and upload it to `CharmKiT/KiT-RT/tools/singularity/`.


## How CharmKiT Works

CharmKiT automates the setup, execution, and result collection for radiative transfer test cases using the KiT-RT solver. The workflow is managed by Python scripts (e.g., `run_hohlraum.py`, `run_lattice.py`) that:

- Define parameter sweeps for each test case (e.g., mesh size, quadrature order, absorption/scattering coefficients).
- Generate the necessary configuration files for KiT-RT.
- Run the KiT-RT solver inside a Singularity container for each parameter combination.
- Collect and save the results (e.g., quantities of interest) as `.npz` files for further analysis.

Scripts support both HPC (SLURM) and local (no-HPC) execution. 


## Running CharmKiT Scripts

CharmKiT provides test-case drivers:

- `run_lattice.py`
- `run_hohlraum.py`

Activate your environment first:

```bash
source venv/bin/activate
```

Both scripts use the same execution flags:

- `--use-slurm`: Submit jobs through SLURM.
- `--use-singularity`: Run KiT-RT through the CPU Singularity image.
- `--cuda`: Run KiT-RT through the CUDA Singularity image (`--nv` is added automatically).
- `--load-from-npz`: Load parameter samples from file (script-dependent behavior).

### Supported run setups

1. **Local mode, raw (no Singularity)**

   ```bash
   python3 run_lattice.py
   # or
   python3 run_hohlraum.py
   ```

   Uses local executable: `./KiT-RT/build/KiT-RT`.

2. **Local mode + Singularity (CPU)**

   ```bash
   python3 run_lattice.py --use-singularity
   # or
   python3 run_hohlraum.py --use-singularity
   ```

   Uses image/executable:
   `KiT-RT/tools/singularity/kit_rt.sif` and `./KiT-RT/build_singularity/KiT-RT`.

3. **Local mode + Singularity + GPU**

   ```bash
   python3 run_lattice.py --cuda
   # or
   python3 run_hohlraum.py --cuda
   ```

   Uses image/executable:
   `KiT-RT/tools/singularity/kit_rt_MPI_cuda.sif` and `./KiT-RT/build_singularity_cuda/KiT-RT`.

4. **SLURM mode, raw (no Singularity)**

   ```bash
   python3 run_lattice.py --use-slurm
   # or
   python3 run_hohlraum.py --use-slurm
   ```

   Generated SLURM scripts call: `srun ./KiT-RT/build/KiT-RT ...`.

5. **SLURM mode + Singularity (CPU)**

   ```bash
   python3 run_lattice.py --use-slurm --use-singularity
   # or
   python3 run_hohlraum.py --use-slurm --use-singularity
   ```

   Generated SLURM scripts call:
   `singularity exec KiT-RT/tools/singularity/kit_rt.sif ./KiT-RT/build_singularity/KiT-RT ...`.

### Not supported

- `--use-slurm --cuda` is intentionally blocked.
  GPU mode is currently supported only for local Singularity runs (no SLURM).


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

If you use CharmKiT or the provided benchmarks in your research, please cite:

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
