# RFdiffusion2 Tutorial: De Novo Enzyme Design from Theozymes

A step-by-step Jupyter Notebook tutorial for designing metallohydrolase enzymes using **RFdiffusion2**. This tutorial accompanies the publications:

> **Computational Design of Metallohydrolases**
> Kim, Woodbury, Ahern et al. *Nature* (2025). [DOI: 10.1038/s41586-025-09746-w](https://doi.org/10.1038/s41586-025-09746-w)
>
> **Atom-level Enzyme Active Site Scaffolding using RFdiffusion2**
> Ahern, Yim, Tischer et al. *Nature Methods* **23**, 96&ndash;105 (2025). [DOI: 10.1038/s41592-025-02975-x](https://doi.org/10.1038/s41592-025-02975-x)

---

## What This Tutorial Covers

This tutorial walks through the complete workflow for designing de novo enzymes around a **theozyme** (theoretical enzyme) — a minimal active-site geometry specifying the catalytic residues, cofactors, and substrate/transition-state analogs needed for a target reaction. RFdiffusion2 then generates full protein scaffolds around this fixed active-site motif.

Two parallel worked examples are provided throughout:

| Example | Source | Metal | Substrate | PDB |
|---------|--------|-------|-----------|-----|
| **Zinc Protease** | Experimental crystal structure | Zn(II) | Phosphonamidate transition-state analog (TSA) | [1QJI](https://www.rcsb.org/structure/1QJI) |
| **Zinc Esterase** | DFT quantum chemistry (Gaussian) | Zn(II) | 4-methylumbelliferyl phosphonate acetate (4MU-PA) | N/A |

---

## Tutorial Workflow Overview

The notebook is organized into three major sections:

### Section I &mdash; Theozyme Creation

Prepare your active-site geometry as a PDB file for RFdiffusion2.

**Option 1: From an Existing PDB Structure**
1. Find a PDB of interest and crop the active-site residues, cofactors, and ligands in PyMOL
2. Clean and format the theozyme PDB with the `prepare_PDB_structure_into_theozyme.py` script (adds REMARK 666 lines for residue tracking, combines ligand/cofactor atoms, removes extraneous hydrogens)
3. *(Optional)* Split the theozyme into subsets of catalytic residues (e.g., HEHH, HEHHY, HEHHYM) to test different design hypotheses in parallel

**Option 2: From a Quantum Chemistry Calculation**
1. Perform a DFT transition-state optimization (e.g., using Gaussian)
2. Extract the optimized geometry from the Gaussian `.log` file into a `.xyz` file
3. Convert the `.xyz` file into a properly formatted PDB using the automated `theozyme_XYZ_to_PDB__MAIN.py` pipeline (identifies residues, builds missing atoms from ideal Rosetta geometries, preserves QM-optimized coordinates)

### Section II &mdash; Theozyme Post-Processing

Prepare the theozyme for scaffold generation.

1. **Add ORI tokens** to specify the desired protein center-of-mass location relative to the active site. The ORI token can be placed manually in PyMOL and the coordinates extracted, then encoded into the theozyme PDB. Options include a single placement or a sphere of multiple placements for broader sampling.
2. *(Optional)* Organize final theozymes into a dedicated `rfdiffusion2_inputs/` folder

### Section III &mdash; Scaffold Generation at Scale

Run RFdiffusion2 to generate protein backbones.

1. **Define fixed atoms and contig maps** &mdash; specify which side-chain atoms should remain fixed during diffusion (e.g., histidine imidazole rings coordinating Zn, glutamate carboxylate acting as general base) and which residues are "unindexed" (sequence position determined by the model)
2. **Auto-generate YAML configs and command files** &mdash; the notebook programmatically creates per-design `.yaml` configuration files and batched command files for all combinations of theozymes, ORI tokens, and protein length ranges
3. **Run RFdiffusion2 inference** &mdash; execute commands individually or submit as SLURM array jobs on HPC
4. **Generate the ligand `.params` file** &mdash; filtering scores each design with PyRosetta, which cannot load `TSA` or `SZD` without a Rosetta parameter file. The notebook builds the command for `Scripts/theozyme_and_ligand_handling/ligands_to_params__UNIFIED.py`, which writes `<LIGAND>.params` and the matching `<LIGAND>.pdb` from the theozyme. This calls Rosetta's `molfile_to_params.py`, which is not bundled here &mdash; set `$ROSETTA` and verify with `python Scripts/repo_paths.py`. Needed once per ligand.
5. **Filter output structures** &mdash; analyze and filter designs by structural quality metrics (radius of gyration, loop content, helix length, ligand burial/SASA, bond-length deviations, catalytic residue geometry, terminus distance)
6. **Visualize and rank** &mdash; plot metric distributions and inspect top-ranked structures in PyMOL

---

## Prerequisites

### 1. Clone the Repository

```bash
git clone --recurse-submodules https://github.com/baker-laboratory/Metallohydrolase_Enzyme_Design.git
cd Metallohydrolase_Enzyme_Design
git submodule update --init --recursive   # nested: fastmpnndesign/lib/LigandMPNN
```

### 2. Create the Conda Environment

All Python dependencies are bundled in a single conda environment:

```bash
conda env create -f Environment/zinc_hydro.yml
conda activate zinc_hydro
```

### 3. Register the Jupyter Kernel

```bash
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=zinc_hydro
```

Then open the notebook and select **zinc_hydro** as the kernel.

### 4. Download the RFdiffusion2 weights (required)

```bash
cd Software/RFdiffusion2 && python setup.py && cd ../..
```

Fetches `rf_diffusion/model_weights/RFD_140.pt` (~1.3 GB) from
`files.ipd.uw.edu`, which is the checkpoint every config in this tutorial
defaults to. Nothing in this repository ships it, and the directory does not
exist until you run this. Re-run as `python setup.py overwrite` if interrupted.

### 5. Download AlphaFold2 Weights (for downstream structure prediction)

```bash
# Requires aria2c: sudo apt install aria2
# Params only (~4 GB). Do NOT use download_all_data.sh -- its argument is a
# directory, and it also pulls BFD/MGnify/PDB70/UniRef90 (~2.6 TB) that
# SuperFold, being single-sequence, never uses.
bash Software/superfold/scripts/download_alphafold_params.sh Software/superfold/alphafold_weights
# alphafold_weights.pth is a TEXT FILE holding the path to the params' parent dir
realpath Software/superfold/alphafold_weights > Software/superfold/alphafold_weights.pth
```

### 6. External Software (used in specific steps)

| Software | Used For | Required? |
|----------|----------|-----------|
| [PyMOL](https://pymol.org/) | Visualizing/cropping PDB structures, placing ORI tokens, inspecting designs | Yes (for manual steps) |
| [Gaussian](https://gaussian.com/) | DFT transition-state optimization | Only for Option 2 (quantum chemistry theozymes) |
| [ChemCraft](https://www.chemcraftprog.com/) | Building initial QM geometries | Only for Option 2 |
| SLURM | HPC job scheduling for batch runs | Optional (can run commands locally) |

---

## Key Dependencies (provided by `zinc_hydro` conda env)

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10 | Runtime |
| PyTorch | 2.3.1 (CUDA 12.1) | Deep learning backend for RFdiffusion2 |
| PyRosetta | 2024.39 | Protein structure manipulation, scoring, residue building |
| OpenBabel | 3.1.1 | Chemical file format conversion (`.xyz` to `.pdb`) |
| NumPy | &mdash; | Numerical computation |
| Pandas | &mdash; | Data analysis and filtering |
| BioPython | 1.84 | Biological sequence/structure utilities |
| RDKit | 2024.03.5 | Cheminformatics |

### Software Submodules (in `Software/`)

| Submodule | Purpose |
|-----------|---------|
| [RFdiffusion2](https://github.com/RosettaCommons/RFdiffusion2) | Generative model for de novo protein backbone design ([Ahern et al., *Nat. Methods* 2025](https://doi.org/10.1038/s41592-025-02975-x)) |
| [PLACER](https://github.com/baker-laboratory/PLACER) | Protein-ligand atomic conformational ensemble prediction |
| [OpenFold](https://github.com/aqlaboratory/openfold) | Structure prediction |
| [SuperFold](https://github.com/rdkibler/superfold) | Structure prediction (AlphaFold2-based) |
| [FastMPNNDesign](https://github.com/ikalvet/fastmpnndesign) | Integrated LigandMPNN + Rosetta sequence design |
| [RFjoint2](https://github.com/RosettaCommons/RFjoint2) | Conditional protein sequence/structure generation |

---

## Directory Structure

```
RFdiffusion2_Tutorial/
|
|-- RFdiffusion2_Tutorial_JupyterNotebook.ipynb   # Main tutorial notebook
|-- README.md                                      # This file
|
|-- theozymes/                         # Example theozyme input structures
|   |-- from_PDB_structure/            # PDB-derived worked example (PDB: 1QJI)
|   |   |-- pdb_00001qji__*.pdb        #   Raw and cropped PDB files
|   |   |-- step1__cleaning/           #   Cleaned/formatted theozymes
|   |   |-- step2__splitting/          #   Theozyme subsets (HEHH, HEHHY, HEHHYM)
|   |   |-- step3__add_ori_tokens/     #   Theozymes with ORI tokens added
|   |   |-- rfdiffusion2_inputs/       #   Final inputs ready for RFdiffusion2
|   |   +-- outputs/                   #   Example RFdiffusion2 output logs
|   |
|   +-- from_quantum_chemistry/        # QM-derived worked example (ZETA_2)
|       |-- Gaussian_TSopt/            #   Gaussian input/output files (.com, .log)
|       |-- *.xyz                      #   Extracted optimized TS geometry
|       |-- pdb_theozymes/             #   Converted PDB theozymes
|       |-- stepFINAL__add_ori_tokens/ #   Theozymes with ORI tokens
|       +-- rfdiffusion2_inputs/       #   Final inputs ready for RFdiffusion2
|
|-- rfd2_configs/                      # 76 auto-generated YAML config files
|-- cmds/                             # Auto-generated command files for RFdiffusion2 runs
|-- slurm_submit/                     # Auto-generated SLURM batch submission scripts
|-- logs/                             # Execution logs
+-- outputs/                          # RFdiffusion2 output structures
```

---

## Quick Start

1. **Activate the environment and open the notebook:**
   ```bash
   conda activate zinc_hydro
   jupyter notebook RFdiffusion2_Tutorial_JupyterNotebook.ipynb
   ```

2. **Run the initialization cell** (Cell 2) &mdash; it locates the repository automatically; there are no paths to edit.

3. **Follow either Option 1 or Option 2** for theozyme creation (Section I). New users should start with **Option 1** (PDB-based), which is simpler and does not require quantum chemistry software.

4. **Post-process theozymes** (Section II) &mdash; add ORI tokens and organize inputs.

5. **Generate and run RFdiffusion2** (Section III) &mdash; the notebook auto-generates all config files and commands. Test a single command first, then submit batch jobs.

6. **Filter and analyze outputs** (Section III.B) &mdash; explore metric distributions and select top designs.

---

## Key Concepts

### Theozyme
A **theozyme** (theoretical enzyme) is a minimal-atom representation of an enzyme active site. It contains only the atoms directly involved in catalysis: metal cofactors (e.g., Zn), substrate/transition-state analogs, and the functional groups of catalytic residues. RFdiffusion2 builds a complete protein scaffold around this fixed core.

### Unindexed Residues
RFdiffusion2 can **unindex** (or "guidepost") motif residues, meaning their sequence position is not pre-specified. The model determines where in the protein sequence each catalytic residue should be placed during inference. This enables a much larger search space than traditional fixed-position approaches.

### Fixed Atoms
For each unindexed residue, users specify which **atoms to fix** during diffusion. For example, fixing the imidazole ring atoms of a histidine coordinating Zn(II) constrains the side-chain geometry while letting RFdiffusion2 freely generate the backbone.

### ORI Tokens
**ORI** (Origin) tokens specify the desired center-of-mass of the generated protein relative to the active site. They guide RFdiffusion2 to build the scaffold in a specific spatial orientation around the theozyme.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `obabel` not found | Ensure the `zinc_hydro` environment is activated. OpenBabel is included in the conda env. |
| RFdiffusion2 fails to run | Verify the `Software/RFdiffusion2` submodule is initialized and model weights are downloaded. Check GPU availability. |
| PyRosetta import errors | Confirm `zinc_hydro` is the active conda env and Jupyter kernel. |
| Theozyme XYZ-to-PDB conversion issues | Enable `keep_temp_files` flags in cell 24 to inspect intermediate PDB files. Check that ligand atom ranges are correct. |
| ORI tokens not visible in PyMOL | Run `show spheres` and adjust scale: `set sphere_scale, 0.5` |
| SLURM jobs fail | Check `logs/` for error output. Verify GPU partition names match your HPC cluster. |

---

## Citation

If you use this tutorial or the associated methods, please cite the primary metallohydrolase paper *and* the RFdiffusion2 method paper:

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

For questions, bug reports, or collaboration inquiries:
- **Seth Woodbury** &mdash; woodbuse@uw.edu
- **Donghyo Kim** &mdash; donghyo@uw.edu
