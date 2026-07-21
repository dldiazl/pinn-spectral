# Temporal Spectral Analysis of Late-Time PINN Error

Code accompanying the manuscript *"Temporal Spectral Analysis of Late-Time
Error in a Physics-Informed Neural Network Solution of the One-Dimensional
Advection-Diffusion Equation"* (submitted to *Mathematics*).

The repository implements, for a single 1D advection-diffusion benchmark
(Pe = 40):

- a closed-form eigenfunction analytical reference,
- three finite-difference baselines (CDS-EF, CDS-CN, Compact-CN),
- a supervised neural-architecture search (NAS) over fully connected networks,
- a physics-informed neural network (PINN) trained with progressive temporal
  windowing, and
- a temporal spectral (DFT-based) diagnostic and low-pass reconstruction of
  the late-time PINN error.

## Pipeline

The workflow is organized in five numbered stages under `scripts/`. Each
stage has a data-generation script (`a`), a postprocessing script (`b`), and
a plotting script (`c`).

| Stage | Scripts | Produces |
|---|---|---|
| 1. Analytical reference | `01a`, `01b`, `01c` | Truncated eigenfunction reference solution and endpoint-convergence diagnostics |
| 2. Finite-difference baselines | `02a`, `02b`, `02c` | CDS-EF, CDS-CN, and Compact-CN solutions and error norms |
| 3. Neural architecture search | `03a`, `03b`, `03c` | NSGA-II/Optuna study and the selected supervised architecture |
| 4. Physics-informed training | `04a`, `04b`, `04c` | Progressively windowed PINN solution and error norms |
| 5. Temporal spectral analysis | `05a`, `05b`, `05c` | Cutoff sensitivity sweep, filtered PINN field, and reference-error comparison |

Run each stage in order from the repository root, for example:

```bat
python scripts\01a_generate_reference_data.py
python scripts\01b_postprocess_reference.py
python scripts\01c_plot_reference.py
python scripts\04a_run_pinn.py
python scripts\05a_generate_spectral_data.py --overwrite
```

Each script accepts `--config` to point to an alternate YAML file and
`--overwrite` to regenerate outputs that already exist.

## Repository layout

```text
configs/     YAML configuration for each pipeline stage
src/         pinn_spectral package: analytical, numerical, NAS, PINN, and
             spectral-analysis implementations
scripts/     Numbered pipeline entry points (data generation, postprocessing,
             plotting)
tests/       Unit and integration tests
data/        Generated datasets (gitignored, except select reference outputs)
results/     Generated figures (tracked), postprocessed tables, metrics, and
             models (gitignored)
```

## Configuration

Each stage reads a dedicated YAML file in `configs/` (`reference.yaml`,
`numerical.yaml`, `nas.yaml`, `pinn.yaml`, `spectral.yaml`) defining the
physical case, grid, and stage-specific parameters. Physical parameters
(`length`, `u_left`, `u_right`, `velocity`, `diffusivity`) and the grid
(`n_space`, `final_time`, `dt`) are repeated across files so each stage can
run independently from its own configuration.

## Setup

The pipeline was run with PyTorch inside a Conda environment on Windows.
Create an environment with the following packages and activate it before
running any script:

```text
numpy
pandas
scipy
matplotlib
torch
optuna
pyyaml
pyarrow
pytest
```

```bat
conda create -n pinn-spectral python=3.11
conda activate pinn-spectral
pip install numpy pandas scipy matplotlib torch optuna pyyaml pyarrow pytest
set PYTHONPATH=src
```

## Tests

```bat
set PYTHONPATH=src
pytest tests
```

`tests/manual_train_architecture.py` is a standalone diagnostic script for
manually training one architecture outside the automated NAS study; it is not
collected by `pytest`.

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
