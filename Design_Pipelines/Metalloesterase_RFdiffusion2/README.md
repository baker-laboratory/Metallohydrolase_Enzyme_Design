# Metallohydrolase Design Pipeline &mdash; Reproduction Tutorial

This directory reproduces the **core computational design pipeline for Design Campaign 1** from the manuscript, as it was provided to peer reviewers. It is the end-to-end workflow that turns a quantum-chemistry-defined active-site geometry (theozyme) into ranked, sequence-designed protein scaffolds ready for wet-lab testing.

This tutorial accompanies:

> **Computational Design of Metallohydrolases**
> Kim, D., Woodbury, S.M., Ahern, W. et al. *Nature* (2025). [DOI: 10.1038/s41586-025-09746-w](https://doi.org/10.1038/s41586-025-09746-w)

The pipeline applies **RFdiffusion2** (Ahern et al., *Nat. Methods* 2025, [DOI: 10.1038/s41592-025-02975-x](https://doi.org/10.1038/s41592-025-02975-x)) to build highly active zinc-dependent hydrolases from theozymes, followed by inpainting, structure prediction, sequence design, and ligand-placement assessment.

> &#x2139;&#xFE0F; **Looking to learn the methodology, not reproduce the manuscript?**
> Start with [`../RFdiffusion2_Tutorial/`](../../RFdiffusion2_Tutorial/) (or [`../RFdiffusion3_Tutorial/`](../../RFdiffusion3_Tutorial/) for the next-generation all-atom model). Those tutorials walk through the same RFdiffusion stage with two fully worked examples and step-by-step guidance.

---

## Pipeline Stages

The notebook [`design_zn_hydrolase.ipynb`](design_zn_hydrolase.ipynb) executes the multi-stage workflow:

1. **RFdiffusion2** &mdash; backbone generation with fixed active-site motifs ([Software/RFdiffusion2/](../../Software/RFdiffusion2))
2. **Scaffold Filtering** &mdash; structural quality assessment (radius of gyration, loop content, ligand burial, bond geometry, catalytic residue placement)
3. **Protein Inpainting (RFjoint2)** &mdash; loop building and structure refinement ([Software/proteininpainting/](../../Software/proteininpainting))
4. **AlphaFold2 Validation (OpenFold / SuperFold)** &mdash; structure prediction to assess designability
5. **Sequence Design (LigandMPNN + FastMPNNDesign)** &mdash; optimized sequence generation around the catalytic pocket
6. **Ligand Placement (PLACER)** &mdash; protein-ligand atomic conformational ensemble prediction

Inputs and intermediate outputs for each stage live alongside the notebook:

```
Design_Pipelines/Metalloesterase_RFdiffusion2/
|-- design_zn_hydrolase.ipynb                # Main pipeline notebook
|-- README.md                                # This file
|-- inputs/
|   |-- invrot/                              # Inverse-rotamer libraries
|   +-- theozymes/                           # Theozyme PDB inputs
|-- working_dir/
|   |-- cmds/                                # Auto-generated batch command files
|   |-- fixed_residue_jsonls/                # Per-design fixed-residue specifications
|   |-- important_dfs/                       # Cached pandas DataFrames (filtering, scoring)
|   +-- rfd2_configs/                        # Auto-generated RFdiffusion2 YAML configs
+-- outputs/
    |-- rfdiffusion2_out/                    # Stage 1: backbone scaffolds
    |-- inpainting_out/                      # Stage 3: inpainted structures
    |-- predesign_out/                       # Pre-MPNN structure preparation
    |-- mpnn_out/                            # Sequence design candidates
    |-- fastMPNNdesign/                      # FastMPNNDesign outputs
    |-- placer_input_structures/             # Inputs prepared for PLACER
    +-- af2_out/                             # AlphaFold2 / SuperFold predictions
```

---

## Setup

This pipeline shares a single conda environment with the rest of the repository.

### 1. Clone the repository (with submodules)

```bash
git clone https://github.com/baker-laboratory/Metallohydrolase_Enzyme_Design.git
cd Metallohydrolase_Enzyme_Design
git submodule update --init --recursive   # nested: fastmpnndesign/lib/LigandMPNN
```

### 2. Create the conda environment

```bash
conda env create -f Environment/zinc_hydro.yml
conda activate zinc_hydro
```

### 3. Register the Jupyter kernel

```bash
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=zinc_hydro
```

### 4. Download the RFdiffusion2 weights (required)

```bash
cd Software/RFdiffusion2 && python setup.py && cd ../..
```

Fetches `rf_diffusion/model_weights/RFD_140.pt` (~1.3 GB) from
`files.ipd.uw.edu`, which is the checkpoint every config in this tutorial
defaults to. Nothing in this repository ships it, and the directory does not
exist until you run this. Re-run as `python setup.py overwrite` if interrupted.

### 5. Download AlphaFold2 weights (required for stage 4)

```bash
# Requires aria2c: sudo apt install aria2
# Params only (~4 GB). Do NOT use download_all_data.sh -- its argument is a
# directory, and it also pulls BFD/MGnify/PDB70/UniRef90 (~2.6 TB) that
# SuperFold, being single-sequence, never uses.
bash Software/superfold/scripts/download_alphafold_params.sh Software/superfold/alphafold_weights
# alphafold_weights.pth is a TEXT FILE holding the path to the params' parent dir
realpath Software/superfold/alphafold_weights > Software/superfold/alphafold_weights.pth
```

### 6. Verify pre-trained model access

This pipeline pulls weights/checkpoints for the following models. Confirm availability before running the corresponding stages:

- [RFdiffusion2](https://github.com/RosettaCommons/RFdiffusion2)
- [LigandMPNN / ProteinMPNN](https://github.com/dauparas/LigandMPNN)
- [PLACER](https://github.com/baker-laboratory/PLACER)
- [OpenFold](https://github.com/aqlaboratory/openfold) / [SuperFold](https://github.com/rdkibler/superfold)
- [RFjoint2 (proteininpainting)](https://github.com/RosettaCommons/RFjoint2)

### 7. Open the notebook

```bash
jupyter notebook Design_Pipelines/Metalloesterase_RFdiffusion2/design_zn_hydrolase.ipynb
```

Select **`zinc_hydro`** as the kernel.

---

## Citation

If you use this pipeline, please cite the primary metallohydrolase paper and the RFdiffusion2 method paper:

```
Kim, D., Woodbury, S.M., Ahern, W. et al.
Computational design of metallohydrolases.
Nature (2025). https://doi.org/10.1038/s41586-025-09746-w

Ahern, W., Yim, J., Tischer, D. et al.
Atom-level enzyme active site scaffolding using RFdiffusion2.
Nature Methods 23, 96-105 (2025). https://doi.org/10.1038/s41592-025-02975-x
```

---

## Contact

- **Seth Woodbury** &mdash; woodbuse@uw.edu
- **Donghyo Kim** &mdash; donghyo@uw.edu
