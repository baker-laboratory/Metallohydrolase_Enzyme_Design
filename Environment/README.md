# Environments

Choose an environment for your task. Run commands from the repository root
unless shown otherwise; [getting started](../Docs/getting-started.md) lists the
workflow entry points.

| Task | Setup | Scope |
|---|---|---|
| Manuscript analysis | [pip/uv](requirements-analysis.txt), [conda](analysis.yml), or pixi `analysis` | CPU; Python 3.10–3.12 |
| Structure preparation, PyRosetta scoring, sequence-design helpers | [zinc_hydro.yml](zinc_hydro.yml) or pixi `design` | Linux x86_64; Python 3.10; CUDA 12.1 |
| Model inference | [Model setup](#model-environments-and-checkpoints) | Separate environments, source code, and checkpoints |

## CPU manuscript analysis

The pip, conda, and pixi specifications cover all four
`Manuscript_Data/*/wetlab_data_analysis.py` apps with the same analysis packages.
Keep `Scripts/` and the manuscript data alongside the apps. No CUDA, models,
or PyRosetta are needed.

### uv or pip

With [uv](https://docs.astral.sh/uv/) on Linux/macOS:

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r Environment/requirements-analysis.txt
.venv/bin/marimo edit Manuscript_Data/Metallohydrolase_Nature_2026/wetlab_data_analysis.py
```

Or with an existing Python 3.11 installation:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r Environment/requirements-analysis.txt
.venv/bin/marimo edit Manuscript_Data/Metallohydrolase_Nature_2026/wetlab_data_analysis.py
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r Environment/requirements-analysis.txt
.\.venv\Scripts\marimo.exe edit Manuscript_Data/Metallohydrolase_Nature_2026/wetlab_data_analysis.py
```

Replace the manuscript path to open another app. Use `marimo run` instead of
`marimo edit` for a read-only interface.

### Conda

```bash
conda env create -f Environment/analysis.yml
conda activate zinc_hydro_analysis
marimo edit Manuscript_Data/Metallohydrolase_Nature_2026/wetlab_data_analysis.py
```

### Pixi

```bash
cd Environment
pixi run -e analysis lab
```

The `lab` task opens the Nature manuscript app. To select another:

```bash
pixi run -e analysis marimo edit ../Manuscript_Data/Metalloprotease_RFdiffusion3_Science_2026/wetlab_data_analysis.py
```

Pixi declares analysis targets for Linux x86_64, Intel/Apple-silicon macOS, and
Windows x86_64. Its first lock may also resolve the licensed Linux helper
environment; use pip or `analysis.yml` if helper-channel access blocks analysis.

## Linux CUDA helper environment

Obtain [PyRosetta access](https://www.pyrosetta.org/downloads) before installing
from the Rosetta channel:

```bash
conda env create -f Environment/zinc_hydro.yml
conda activate zinc_hydro
python Scripts/env_config.py
python -c "import pyrosetta, torch; print('helper imports OK'); print('CUDA available:', torch.cuda.is_available())"
obabel -V
```

The specification retains Python 3.10, PyTorch 2.3.1, CUDA 12.1, and PyRosetta
2024.39, plus JupyterLab, marimo, Open Babel, and structure/chemistry packages.
The equivalent pixi environment is named `design`:

```bash
cd Environment
pixi run -e design python -c "import pyrosetta, torch; print('helper imports OK')"
```

Scripts that import model modules such as `rf_diffusion` or `openfold`, or call
Foldseek, still need those tools installed separately. LigandMPNN helpers need
their source checkout and weights. A helper import check does not test inference.

GPU helpers require a host NVIDIA driver; conda and container CUDA packages do
not install it. No complete CPU, AMD/ROCm, or Apple-silicon design environment is
provided.

## Model environments and checkpoints

Initialize the required submodules and use each model's own interpreter or
upstream container. The pinned [RFdiffusion2 metadata](https://github.com/RosettaCommons/RFdiffusion2/blob/d365cbf4db3958814a9f8e4f6f94fa309dfebc2b/pyproject.toml)
requires Python ≥3.11; [foundry](https://github.com/RosettaCommons/foundry/blob/aad357b776e3c0d6b973080f8f8c4bcf3ed21e40/pyproject.toml)
and [AlphaFold3](https://github.com/google-deepmind/alphafold3/blob/c0f97eda2f1f482fd94d3a38bece18c7069b4a5c/pyproject.toml)
require ≥3.12, beyond the helper environment's Python 3.10.

| Component | Setup reference |
|---|---|
| RFdiffusion2 | [Pinned README](https://github.com/RosettaCommons/RFdiffusion2/blob/d365cbf4db3958814a9f8e4f6f94fa309dfebc2b/README.md): environment/container and weights |
| RFdiffusion3 through foundry | [Pinned README](https://github.com/RosettaCommons/foundry/blob/aad357b776e3c0d6b973080f8f8c4bcf3ed21e40/README.md): package environment and checkpoints |
| LigandMPNN/FastMPNNDesign, SuperFold/AF2, AF3, PLACER | [Software guide](../Docs/software-and-models.md): source, dependencies, weights, databases, and access |

Configure notebook/script launchers for those installations. The helper
containers below do not include model runtimes or checkpoints.

## Optional helper containers

Both recipes install the helper manifest under `/opt/conda/envs/zinc_hydro`.
The repository and model data must be mounted at runtime.

### Apptainer

Build on Linux x86_64 from the repository root:

```bash
apptainer build Environment/zinc_hydro.sif Environment/zinc_hydro.def
apptainer run Environment/zinc_hydro.sif Scripts/env_config.py
export ZINC_HYDRO_SIF="$PWD/Environment/zinc_hydro.sif"
```

Building may require `--fakeroot` or an administrator. The runscript executes
helper Python. Bind the repository if your site does not bind the working
directory automatically. GPU helpers need
`apptainer exec --nv IMAGE.sif python SCRIPT.py`; `ZINC_HYDRO_SIF` alone adds
neither `--nv` nor data mounts.

### Docker

```bash
docker build --platform linux/amd64 -f Environment/Dockerfile -t zinc_hydro:local .
docker run --rm -it -v "$PWD":/work -w /work zinc_hydro:local bash
```

Add `--gpus all` for GPU helpers with NVIDIA Container Toolkit configured on the
host. Mount additional data as needed. The root `.dockerignore` restricts build
context to the recipe and environment specification. Package licenses still
apply inside containers.

## Helper configuration

[`Scripts/env_config.py`](../Scripts/env_config.py) reads these optional settings;
model packages use their own configuration.

| Variable | Default | Purpose |
|---|---|---|
| `ZINC_HYDRO_SIF` | Active Python | Interpreter or executable Apptainer image for helper subprocesses |
| `ZINC_HYDRO_OBABEL` | `obabel` on `PATH` | Open Babel executable |
| `ZINC_HYDRO_REPO` | Location of `env_config.py`, then an upward directory search | Repository root override |

Inspect resolved paths with `python Scripts/env_config.py`. Missing PyRosetta
and Open Babel are expected in an analysis-only environment.

## Reproducibility and validation

These specifications resolve packages at installation time; no `pixi.lock` is
shipped. Save resolved versions with your outputs. Helper/container installs
also fetch unpinned ProDy from Git and depend on channel/license access.

[`zinc_hydro.lock.yml`](zinc_hydro.lock.yml) preserves the historical Linux conda
export unchanged. Its Git dependencies are unpinned, and it omits checkpoints,
databases, and external tools: it is provenance, not a complete reproduction
lock.

A clean Linux Python 3.11 analysis installation was checked. Conda/pixi helper
solves, container builds, GPU inference, and other operating systems were not
validated during this refresh.
