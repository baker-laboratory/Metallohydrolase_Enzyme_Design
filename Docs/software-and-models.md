# Software and model weights

[← Documentation](README.md) · [Environment setup](../Environment/README.md)

Design workflows need separate helper environments, model code, and weights.
Install the tools your workflow uses; wet lab analysis needs none of these
model checkouts.

## Software checkouts

From the repository root:

```bash
git submodule update --init --recursive
git submodule status --recursive
```

These commands use the recorded commits, including Foundry despite its
`production` branch setting. Avoid `--remote` for reproduction: it changes
versions. Record parent/submodule commits, the environment, and checkpoint IDs.

| Checkout | Role | Installation reference |
| :--- | :--- | :--- |
| `Software/RFdiffusion2/` | Active-site scaffold generation | [Bundled README](https://github.com/RosettaCommons/RFdiffusion2/blob/d365cbf4db3958814a9f8e4f6f94fa309dfebc2b/README.md) |
| `Software/foundry/` | RFdiffusion3 via `rfd3 design` | [Foundry README](https://github.com/RosettaCommons/foundry/blob/aad357b776e3c0d6b973080f8f8c4bcf3ed21e40/README.md) and [RFdiffusion3 guide](https://github.com/RosettaCommons/foundry/blob/aad357b776e3c0d6b973080f8f8c4bcf3ed21e40/models/rfd3/README.md) |
| `Software/fastmpnndesign/` | LigandMPNN and Rosetta sequence design | [Bundled README](https://github.com/ikalvet/fastmpnndesign/blob/18ede0bf662ab8f194872a0911d2e50864710a5e/README.md); includes nested LigandMPNN |
| `Software/PLACER/` | Protein–ligand conformational ensembles | [Bundled README](https://github.com/baker-laboratory/PLACER/blob/7dc5563300e7143582d85636bdd413d4c7688e9a/README.md) |
| `Software/superfold/` | AlphaFold2-based validation | [Bundled README](https://github.com/rdkibler/superfold/blob/021d2d1d0501cf67c83039d18225ed3eab0bcefa/README.md) |
| `Software/openfold/` | OpenFold source used by prediction tools | [Bundled README](https://github.com/aqlaboratory/openfold/blob/be2ec1841f16c966c65ae0e7599ebbadc725757d/README.md) |
| `Software/proteininpainting/` | RFjoint2 inpainting | [Bundled README](https://github.com/RosettaCommons/RFjoint2/blob/c1603f8c36165a408d1ce9449aeba7dc976421a5/README.md) |
| `Software/alphafold3/` | AlphaFold3 validation | [Installation guide](https://github.com/google-deepmind/alphafold3/blob/c0f97eda2f1f482fd94d3a38bece18c7069b4a5c/docs/installation.md) |
| `Software/foldseek/` | Structural search and clustering | [Bundled README](https://github.com/steineggerlab/foldseek/blob/e28c8d2045b2603f192f20ffdc81b6084b53cc5d/README.md) |

[`Software/theozyme_XYZ_2_PDB__beta/`](../Software/theozyme_XYZ_2_PDB__beta/)
is a bundled PyRosetta/Open Babel converter that reconstructs residues around
quantum-chemistry coordinates.

Follow each tool's setup guide and configure its notebook launcher. RFdiffusion2
and Foundry require newer Python than the shared 3.10 helper environment.

## Model weights

Keep downloaded weights in ignored storage such as `model_weights/`, a model
cache, or the tool's expected location. The parent repository contains no
checkpoints; **PLACER's upstream submodule includes its own weights** under
`Software/PLACER/weights/`.

For AlphaFold3, follow the [parameter access instructions](https://github.com/google-deepmind/alphafold3/blob/c0f97eda2f1f482fd94d3a38bece18c7069b4a5c/docs/model_parameters.md).
`AF3_RUNNER` selects a launcher, not a weight file.

### RFdiffusion2

Follow the upstream setup guide, then run from the repository root:

```bash
(
    cd Software/RFdiffusion2
    python setup.py
)
```

This downloads multiple containers and checkpoints, including `RFD_140.pt`
and `RFD_173.pt`; it is not a package installer or a single-checkpoint download.
Allow space for all downloads. Tutorial configs use
`rf_diffusion/model_weights/RFD_140.pt`; follow the README above for inference.

### RFdiffusion3

After installing Foundry in its model environment:

```bash
foundry install rfd3
foundry list-installed
```

Foundry searches `~/.foundry/checkpoints/` and colon-separated
`FOUNDRY_CHECKPOINT_DIRS`. Tutorial `checkpoint_path` becomes CLI `ckpt_path`.
Record the checkpoint: registry defaults can change independently of source pins.

### LigandMPNN

From the repository root:

```bash
git submodule update --init --recursive Software/fastmpnndesign
bash Software/fastmpnndesign/lib/LigandMPNN/get_model_params.sh model_weights/
export LIGANDMPNN_WEIGHTS="$PWD/model_weights"
```

`LIGANDMPNN_DIR` locates code; `LIGANDMPNN_WEIGHTS` locates weights. Match the
driver's checkpoint setting: metalloesterase uses
`Software/fastmpnndesign/lib/LigandMPNN/model_params/`; phosphotriesterase supports
repository-level `model_weights/`. Adjust the path or download destination.

### AlphaFold2 / SuperFold

The supplied parameter-only downloader requires `aria2c`:

```bash
bash Software/superfold/scripts/download_alphafold_params.sh Software/superfold/alphafold_weights
python - <<'PY'
from pathlib import Path
weights = Path('Software/superfold/alphafold_weights').resolve()
Path('Software/superfold/alphafold_weights.pth').write_text(str(weights) + '\n')
PY
```

The `.pth` file must name the directory **containing** `params/`. These
single-sequence workflows need the parameter download; the full database
downloader is separate and much larger.

## External tools and local configuration

| Tool or setting | Used for |
| :--- | :--- |
| Rosetta `molfile_to_params.py` | New ligand parameters; set `ROSETTA` or `MOLFILE_TO_PARAMS` |
| `ROSETTA_RESIDUE_TYPES` | Residue-type database override |
| Open Babel | Format conversion; `ZINC_HYDRO_OBABEL` or `PATH` |
| Gaussian / ChemCraft | New quantum-chemistry theozymes; unnecessary for inspecting deposited geometries |
| PyMOL | Structure inspection and motif preparation |
| SLURM | Batch scheduling; adapt resources, account, paths, and environment to your cluster |

Check paths with `python Scripts/env_config.py` and
`python Scripts/repo_paths.py`. The latter also accepts `OBABEL`; keep it
consistent with `ZINC_HYDRO_OBABEL`. See the [environment guide](../Environment/README.md)
for containers and configuration.

## Licensing

The [MIT License](../LICENSE) covers original project code. Third-party code,
executables, model parameters, and external software retain their own terms.

| Component | Terms and attribution |
| :--- | :--- |
| PyRosetta | [PyRosetta license](https://github.com/RosettaCommons/rosetta/blob/main/LICENSE.PyRosetta.md) and [licensing portal](https://els2.comotion.uw.edu/product/pyrosetta) |
| AlphaFold3 parameters and outputs | [Google DeepMind parameter terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md); distinct from the source-code license |
| Foldseek | [Upstream GPL-3.0 license](https://github.com/steineggerlab/foldseek/blob/e28c8d2045b2603f192f20ffdc81b6084b53cc5d/LICENSE.md) |
| `Scripts/enzyme_design/DAlphaBall.gcc` | Rosetta-distributed third-party executable; see [upstream source](https://github.com/RosettaCommons/rosetta/tree/main/source/external/DAlpahBall) and [Rosetta licensing](https://github.com/RosettaCommons/rosetta/blob/main/LICENSE.md) |
| Patched LigandMPNN side-chain utilities | [Source and retained copyright](../Scripts/mpnn_relevant_utils/ligandmpnn_patched/sc_utils.py) |

Cite the [project publications](citations.md) and the method papers requested by
the upstream tools you use.
