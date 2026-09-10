# Environment

Four ways to get a working environment. Pick one — you do not need all of them.

## Which one do I want?

| I want to… | Use | Time | Needs a GPU? |
|---|---|---|---|
| Re-plot the wet lab data in `Manuscript_Data/` | [`analysis.yml`](analysis.yml) or [`requirements-analysis.txt`](requirements-analysis.txt) | ~1 min | No |
| Run the RFdiffusion2/3 design tutorials and pipelines | [`zinc_hydro.yml`](zinc_hydro.yml) | 20–40 min | Yes, for inference |
| Reproduce our exact package versions | [`zinc_hydro.lock.yml`](zinc_hydro.lock.yml) | 20–40 min | Yes, for inference |
| Work on a cluster that only allows containers | [`zinc_hydro.def`](zinc_hydro.def) (Apptainer) | ~40 min to build | Yes, for inference |
| Use Docker instead of Apptainer | [`Dockerfile`](Dockerfile) | ~40 min to build | Yes, for inference |
| Avoid conda entirely, with a real lockfile | [`pixi.toml`](pixi.toml) | ~5–40 min | Depends on env |

---

## 1. Analysis only (smallest, works everywhere)

Everything needed for `Manuscript_Data/*/wetlab_data_analysis.ipynb` and nothing
else. No CUDA, no PyTorch, no PyRosetta, no license.

```bash
conda env create -f Environment/analysis.yml
conda activate zinc_hydro_analysis
python -m ipykernel install --user --name=zinc_hydro_analysis
```

Or with `uv` / `pip`:

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -r Environment/requirements-analysis.txt
python -m ipykernel install --user --name=zinc_hydro_analysis
```

## 2. Full design environment

```bash
conda env create -f Environment/zinc_hydro.yml
conda activate zinc_hydro
python -m ipykernel install --user --name=zinc_hydro
```

`zinc_hydro.yml` pins minor versions only, so conda can pick builds that suit
your platform and drivers. For our exact builds — every package pinned to its
build string, linux-64 only — use `zinc_hydro.lock.yml` instead. Prefer the
lockfile when you are checking a published number; prefer the portable file when
you just need it to install.

**PyRosetta needs a license** (free for academic use). Register at
[els2.comotion.uw.edu/product/pyrosetta](https://els2.comotion.uw.edu/product/pyrosetta);
the `conda.rosettacommons.org` channel then serves the builds.

**CPU-only?** Delete the deep learning block from `zinc_hydro.yml` (`pytorch`
through `tensorflow`). Everything except RFdiffusion2/3 inference still works —
including the theozyme XYZ→PDB conversion and all PyRosetta scoring.

## 3. Containers

```bash
cd Environment
apptainer build zinc_hydro.sif zinc_hydro.def
export ZINC_HYDRO_SIF=$PWD/zinc_hydro.sif
```

or, with Docker (run from the repository root):

```bash
docker build -f Environment/Dockerfile -t zinc_hydro:latest .
```

The image's runscript is the environment's python, so `./zinc_hydro.sif script.py`
works anywhere this repository expects an interpreter. Pass `--nv` for GPU work.

> **A container is optional.** The conda environment above provides everything
> the scripts need, including PyRosetta and Open Babel. Build an image only if
> your cluster requires one, or if you want a single portable artifact.

## 4. pixi

```bash
cd Environment
pixi run -e analysis jupyter lab      # analysis stack, cross-platform
pixi run -e design python -c "import pyrosetta"   # full stack, linux-64
```

`pixi install` writes a `pixi.lock` that reproduces the solve across machines.
Commit that lock if you want a fixed environment; we ship the manifest without
one so the first install resolves against current channels.

---

## Environment variables

Every script and notebook here reads the same three variables. All are optional
— the defaults are correct inside the conda environment.

| Variable | Default when unset | Purpose |
|---|---|---|
| `ZINC_HYDRO_SIF` | the active python interpreter | Apptainer image used to run helper scripts |
| `ZINC_HYDRO_OBABEL` | `obabel` from `$PATH` | Open Babel executable |
| `ZINC_HYDRO_REPO` | auto-detected by walking up from the working directory | Repository root |

Check what will actually be used before running anything:

```bash
python Scripts/env_config.py
```

```
### ENVIRONMENT REPORT ###
    python            : /opt/conda/envs/zinc_hydro/bin/python
    conda_env         : zinc_hydro
    repo_root         : /home/you/Metallohydrolase_Enzyme_Design
    runner            : /opt/conda/envs/zinc_hydro/bin/python
    obabel            : /opt/conda/envs/zinc_hydro/bin/obabel
    ...
```

Any line prefixed with `!!` is a problem, with the fix printed underneath.

---

## Files

| File | What it is |
|---|---|
| `analysis.yml` | Conda env for the wet lab analysis notebooks only |
| `requirements-analysis.txt` | The same, for `pip` / `uv` |
| `zinc_hydro.yml` | Portable conda env for the full design stack (minor-version pins) |
| `zinc_hydro.lock.yml` | Exact conda export of the environment we ran (linux-64, build strings) |
| `zinc_hydro.def` | Apptainer/Singularity definition, built from `zinc_hydro.yml` |
| `Dockerfile` | Docker equivalent of `zinc_hydro.def` |
| `pixi.toml` | pixi manifest with `analysis` and `design` environments |
