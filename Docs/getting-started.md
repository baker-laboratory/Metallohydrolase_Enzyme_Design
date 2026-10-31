# Getting started

[← Documentation](README.md)

Start with a CPU analysis app. Design workflows require additional model
installations, checkpoints, and tools.

## Get the repository

```bash
git clone https://github.com/baker-laboratory/Metallohydrolase_Enzyme_Design.git
cd Metallohydrolase_Enzyme_Design
```

Run commands from this root. Analysis uses `Manuscript_Data/` and `Scripts/`;
software submodules are needed only for design workflows.

## Install analysis dependencies

Choose Conda or uv:

```bash
conda env create -f Environment/analysis.yml
conda activate zinc_hydro_analysis
```

```bash
uv venv .venv --python 3.11
uv pip install --python .venv -r Environment/requirements-analysis.txt
source .venv/bin/activate
```

The activation command above is for Bash/Zsh. On Windows PowerShell, use
`.venv\Scripts\Activate.ps1`. For pip, Pixi, and platform details, see the
[environment guide](../Environment/README.md).

## Open an app

```bash
marimo edit Manuscript_Data/Metallohydrolase_Nature_2026/wetlab_data_analysis.py
```

Each of the [four datasets](data-and-reproduction.md) has its own
`wetlab_data_analysis.py`. Change the directory to select another publication.
The app opens in your browser; no GPU or model weights are required.
Read the dataset's export settings before running: some apps write figures
into tracked directories.

## Start a design workflow

1. Choose a [tutorial or campaign](workflows.md).
2. Run `git submodule update --init --recursive`.
3. Install the [helper environment](../Environment/README.md) and the selected
   model's separate [runtime and weights](software-and-models.md).
4. Configure notebook launchers, output paths, and scheduler settings. Inspect
   one generated command before launching a batch.

Check helper and ligand tooling in the intended environment:

```bash
python Scripts/env_config.py
python Scripts/repo_paths.py
```

These reports check path resolution, not model inference. Missing PyRosetta,
Open Babel, and Rosetta tools is expected in the analysis-only environment.

## Troubleshooting

| Symptom | Check |
| :--- | :--- |
| `marimo` is missing | Activate the analysis environment |
| Excel imports fail | Confirm `openpyxl` is installed |
| A software directory is empty | Initialize its submodule recursively |
| `rfd3` is missing | Activate the Foundry model environment |
| Ligand parameterization fails | Inspect `python Scripts/repo_paths.py` |
| A job cannot find tools or inputs | Check environment activation, paths, container mounts, and scheduler settings |

For help, [open an issue](https://github.com/baker-laboratory/Metallohydrolase_Enzyme_Design/issues)
with the command, traceback, repository commit, and environment details.
