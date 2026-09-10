# RFdiffusion3 Tutorial: De Novo Enzyme Design from Theozymes

A step-by-step Jupyter Notebook tutorial for designing metallohydrolase enzymes using **RFdiffusion3** (via the [foundry](https://github.com/RosettaCommons/foundry) framework). This tutorial accompanies the publications:

> **Computational Design of Metallohydrolases**
> Kim, Woodbury, Ahern et al. *Nature* (2025). [DOI: 10.1038/s41586-025-09746-w](https://doi.org/10.1038/s41586-025-09746-w)
>
> **De novo Design of All-atom Biomolecular Interactions with RFdiffusion3**
> Butcher, Krishna, Mitra et al. *bioRxiv* (2025). [DOI: 10.1101/2025.09.18.676967](https://doi.org/10.1101/2025.09.18.676967) &mdash; *preprint, citation will update on peer-reviewed publication.*
>
> **Atom-level Enzyme Active Site Scaffolding using RFdiffusion2**
> Ahern, Yim, Tischer et al. *Nature Methods* **23**, 96&ndash;105 (2025). [DOI: 10.1038/s41592-025-02975-x](https://doi.org/10.1038/s41592-025-02975-x) &mdash; *for context on the predecessor method that Sections I and II were originally designed for.*

---

## What This Tutorial Covers

This tutorial walks through the complete workflow for designing de novo enzymes around a **theozyme** (theoretical enzyme) using **RFdiffusion3**, the next-generation all-atom generative model for protein design. Sections I and II (theozyme creation and post-processing) are identical to the [RFdiffusion2 Tutorial](../RFdiffusion2_Tutorial/), while Section III demonstrates the RFdiffusion3-specific scaffold generation and filtering pipeline.

Two parallel worked examples are provided throughout:

| Example | Source | Metal | Substrate | PDB |
|---------|--------|-------|-----------|-----|
| **Zinc Protease** | Experimental crystal structure | Zn(II) | Phosphonamidate transition-state analog (TSA) | [1QJI](https://www.rcsb.org/structure/1QJI) |
| **Zinc Esterase** | DFT quantum chemistry (Gaussian) | Zn(II) | 4-methylumbelliferyl phosphonate acetate (4MU-PA) | N/A |

---

## Key Differences from the RFdiffusion2 Tutorial

| Feature | RFdiffusion2 | RFdiffusion3 |
|---------|--------------|--------------|
| **Config format** | YAML (`.yaml`) | JSON (`.json`) |
| **Inference framework** | `RFdiffusion2/rf_diffusion/run_inference.py` | the open-source `rfd3 design` CLI from foundry |
| **Config dialect** | N/A | `dialect: 2` |
| **Conditioning** | Radius of gyration, relative SASA | Atomwise RASA, atomwise H-bonds, classifier-free guidance |
| **Sampling parameters** | Timesteps, deterministic flag | `cfg_scale`, `step_scale`, `gamma_0`, `gamma_min`, `s_jitter_origin` |
| **Output format** | PDB files | CIF.gz + per-structure JSON metrics |
| **Filtering** | Single-stage (PyRosetta `.sc` files) | Two-stage: rapid JSON-based + PyRosetta-based |
| **Post-processing** | N/A | Geometry idealization (Cartesian relax) |

---

## Release status

foundry is tracked at its `production` branch, currently **`v0.2.0`**. The whole
tutorial runs today, end to end.

The Section III and IV helper scripts are bundled in this repository:

| Step | Script |
|---|---|
| Output processing and scoring | `Scripts/process_diffusion3_outputs.py` |
| Catalytic-residue sequence-distance filter | `Scripts/scaffold_handling/filter_pdbsDIR_by_catres_sequence_distance.py` |
| Ligand parameterization | `Scripts/theozyme_and_ligand_handling/ligands_to_params__UNIFIED.py` |
| Geometry idealization | `Scripts/scaffold_handling/idealize_rfdiffusion3_geometry.py` |

The scaffold-analysis and idealization steps need only the `zinc_hydro` environment. Ligand parameterization additionally calls Rosetta's `molfile_to_params.py`, which is not bundled here — set `$ROSETTA` and check with `python Scripts/repo_paths.py`. The remaining scripts — the idealizer and the output
processor use PyRosetta, and the sequence-distance filter is pure standard
library.

---

## Tutorial Workflow Overview

The notebook is organized into four major sections:

### Section I &mdash; Theozyme Creation

Prepare your active-site geometry as a PDB file. *Identical to the RFdiffusion2 tutorial.*

**Option 1: From an Existing PDB Structure**
1. Find a PDB of interest and crop the active-site residues, cofactors, and ligands in PyMOL
2. Clean and format the theozyme PDB with the `prepare_PDB_structure_into_theozyme.py` script
3. *(Optional)* Split the theozyme into subsets of catalytic residues

**Option 2: From a Quantum Chemistry Calculation**
1. Perform a DFT transition-state optimization (e.g., using Gaussian)
2. Extract the optimized geometry from the Gaussian `.log` file into a `.xyz` file
3. Convert the `.xyz` file into a properly formatted PDB

### Section II &mdash; Theozyme Post-Processing

Prepare the theozyme for scaffold generation. *Identical to the RFdiffusion2 tutorial.*

1. **Add ORI tokens** to specify the desired protein center-of-mass location
2. *(Optional)* Organize final theozymes into a dedicated input folder

### Section III &mdash; RFdiffusion3 Scaffold Generation at Scale

Run RFdiffusion3 to generate protein backbones. *This is the RFdiffusion3-specific section.*

1. **Define fixed atoms and unindexed residues** &mdash; specify which side-chain atoms should remain fixed and which residues are unindexed (sequence position determined by the model)
2. **Generate JSON config files** &mdash; programmatically create per-design `.json` configuration files with RFd3 dialect 2 format, supporting atomwise RASA conditioning, atomwise H-bond conditioning, and flexible grouping of design variants
3. **Build inference commands** &mdash; generate Apptainer-based commands with hyperparameter sweeping (classifier-free guidance, cfg scale, step scale, gamma parameters, jitter)
4. **Submit batch jobs** &mdash; execute commands individually or submit as SLURM array jobs on HPC
5. **Rapid JSON-based filtering** &mdash; parse per-structure JSON metrics and filter by structural quality (insertion RMSD, chainbreaks, loop fraction, clashes, SASA, radius of gyration)
6. **Additional scaffold filters** &mdash; run PyRosetta-based analysis and filter designs by bond-length deviations, side-chain quality, ligand burial, and terminus distance

### Section IV &mdash; Geometry Idealization (Optional)

Post-process filtered scaffolds with Cartesian relaxation to idealize bond geometry.

1. **Cartesian relax** &mdash; constrained energy minimization to fix bond-length and bond-angle deviations while preserving the designed backbone
2. **Filter idealized structures** &mdash; verify geometry quality after idealization

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

### 4. RFdiffusion3 Setup

Inference runs through the `rfd3 design` command-line interface from
[foundry](https://github.com/RosettaCommons/foundry), included as a submodule at
`Software/foundry/` (branch `production`). Install it once:

```bash
git submodule update --init --recursive Software/foundry
pip install -e "Software/foundry[rfd3]"     # or: pip install "rc-foundry[rfd3]"
foundry install rfd3                        # fetches the checkpoint
```

`foundry install rfd3` writes the checkpoint to `~/.foundry/checkpoints`, which
foundry discovers automatically — so the notebook leaves `checkpoint_path` unset
by default. To keep checkpoints elsewhere, pass `--checkpoint-dir` and list that
directory in `FOUNDRY_CHECKPOINT_DIRS` (colon-separated).

A GPU is required. See `Software/foundry/models/rfd3/README.md` for the full
option reference.

### 5. External Software (used in specific steps)

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
| PyTorch | 2.3.1 (CUDA 12.1) | Deep learning backend |
| PyRosetta | 2024.39 | Protein structure manipulation, scoring, residue building |
| OpenBabel | 3.1.1 | Chemical file format conversion (`.xyz` to `.pdb`) |
| NumPy | &mdash; | Numerical computation |
| Pandas | &mdash; | Data analysis and filtering |
| BioPython | 1.84 | Biological sequence/structure utilities |
| RDKit | 2024.03.5 | Cheminformatics |

### Software Submodules (in `Software/`)

| Submodule | Purpose |
|-----------|---------|
| [foundry](https://github.com/RosettaCommons/foundry) | RFdiffusion3 &mdash; all-atom generative model for de novo protein design ([Butcher et al., *bioRxiv* 2025](https://doi.org/10.1101/2025.09.18.676967), preprint) |
| [RFdiffusion2](https://github.com/RosettaCommons/RFdiffusion2) | Previous-generation backbone design model ([Ahern et al., *Nat. Methods* 2025](https://doi.org/10.1038/s41592-025-02975-x)) |
| [PLACER](https://github.com/baker-laboratory/PLACER) | Protein-ligand atomic conformational ensemble prediction |
| [OpenFold](https://github.com/aqlaboratory/openfold) | Structure prediction |
| [SuperFold](https://github.com/rdkibler/superfold) | Structure prediction (AlphaFold2-based) |
| [FastMPNNDesign](https://github.com/ikalvet/fastmpnndesign) | Integrated LigandMPNN + Rosetta sequence design |
| [RFjoint2](https://github.com/RosettaCommons/RFjoint2) | Conditional protein sequence/structure generation |

---

## Directory Structure

```
RFdiffusion3_Tutorial/
|
|-- RFdiffusion3_Tutorial_JupyterNotebook.ipynb   # Main tutorial notebook
|-- README.md                                      # This file
|
|-- theozymes/                         # Example theozyme input structures
|   |-- from_PDB_structure/            # PDB-derived worked example (PDB: 1QJI)
|   +-- from_quantum_chemistry/        # QM-derived worked example (ZETA_2)
|
|-- rfd3_json/                         # Auto-generated JSON config files (RFd3 format)
|-- cmds/                              # Auto-generated command files for RFd3 runs
|-- slurm_submit/                      # Auto-generated SLURM batch submission scripts
|-- logs/                              # Execution logs
+-- outputs/                           # RFdiffusion3 output structures
```

---

## Quick Start

1. **Activate the environment and open the notebook:**
   ```bash
   conda activate zinc_hydro
   jupyter notebook RFdiffusion3_Tutorial_JupyterNotebook.ipynb
   ```

2. **Run the initialization cell** (Cell 2) &mdash; set `github_repo_dir` to your local clone path.

3. **Follow either Option 1 or Option 2** for theozyme creation (Section I). New users should start with **Option 1** (PDB-based), which is simpler and does not require quantum chemistry software.

4. **Post-process theozymes** (Section II) &mdash; add ORI tokens and organize inputs.

5. **Generate and run RFdiffusion3** (Section III) &mdash; the notebook generates JSON config files and Apptainer commands. Test a single command first, then submit batch jobs.

6. **Filter outputs** (Section III.B-C) &mdash; first apply rapid JSON-based filters, then run PyRosetta-based analysis and additional filtering.

7. **Geometry idealization** (Section IV, optional) &mdash; Cartesian relax to improve bond geometry of top designs.

---

## Key Concepts

### Theozyme
A **theozyme** (theoretical enzyme) is a minimal-atom representation of an enzyme active site. It contains only the atoms directly involved in catalysis: metal cofactors (e.g., Zn), substrate/transition-state analogs, and the functional groups of catalytic residues. RFdiffusion3 builds a complete protein scaffold around this fixed core.

### Unindexed Residues
RFdiffusion3 can **unindex** motif residues, meaning their sequence position is not pre-specified. The model determines where in the protein sequence each catalytic residue should be placed during inference. This enables a much larger search space than traditional fixed-position approaches.

### Fixed Atoms
For each unindexed residue, users specify which **atoms to fix** during diffusion. For example, fixing the imidazole ring atoms of a histidine coordinating Zn(II) constrains the side-chain geometry while letting RFdiffusion3 freely generate the backbone.

### ORI Tokens
**ORI** (Origin) tokens specify the desired center-of-mass of the generated protein relative to the active site. They guide RFdiffusion3 to build the scaffold in a specific spatial orientation around the theozyme.

### Classifier-Free Guidance (CFG)
RFdiffusion3 supports **classifier-free guidance**, a technique that steers generation toward structures that better satisfy specified conditions (e.g., RASA, H-bonds). The `cfg_scale` parameter controls the strength of this guidance.

### Atomwise RASA Conditioning
RFdiffusion3 can condition on **per-atom relative accessible surface area (RASA)**, allowing fine-grained control over which ligand/substrate atoms should be buried, partially buried, or solvent-exposed in the designed scaffold.

### Atomwise H-Bond Conditioning
Similarly, **atomwise H-bond conditioning** specifies desired hydrogen-bonding interactions between the protein and specific ligand atoms, enabling explicit design of catalytic hydrogen-bond networks.

### Geometry Idealization
RFdiffusion3 outputs may contain minor bond-geometry deviations. A **geometry idealization** step (Cartesian relaxation with tight coordinate constraints) fixes these deviations while preserving the designed backbone conformation.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `obabel` not found | Ensure the `zinc_hydro` environment is activated. OpenBabel is included in the conda env. |
| `rfd3: command not found` | Install foundry into the active environment: `pip install -e "Software/foundry[rfd3]"`. |
| Checkpoint not found | Run `foundry install rfd3`. If the checkpoint lives outside `~/.foundry/checkpoints`, set `FOUNDRY_CHECKPOINT_DIRS` or `checkpoint_path` in the inference cell. |
| JSON config errors | Verify that all required keys (`input`, `ligand`, `length`, `unindex`, `select_fixed_atoms`, `dialect`) are present in the generated JSON files. |
| PyRosetta import errors | Confirm `zinc_hydro` is the active conda env and Jupyter kernel. |
| Theozyme XYZ-to-PDB conversion issues | Enable `keep_temp_files` flags in the conversion cell to inspect intermediate PDB files. Check that ligand atom ranges are correct. |
| ORI tokens not visible in PyMOL | Run `show spheres` and adjust scale: `set sphere_scale, 0.5` |
| SLURM jobs fail | Check `logs/` for error output. Verify GPU partition names match your HPC cluster. |
| Geometry idealization diverges | Try reducing `coord_cst_stdev` (tighter constraints) or increasing `coord_cst_weight`. |

---

## Citation

If you use this tutorial or the associated methods, please cite the primary metallohydrolase paper *and* the RFdiffusion3 method paper. If you also use Sections I and II conventions inherited from the RFdiffusion2 workflow, please cite the RFdiffusion2 method paper as well:

```
Kim, D., Woodbury, S.M., Ahern, W. et al.
Computational design of metallohydrolases.
Nature (2025). https://doi.org/10.1038/s41586-025-09746-w

Butcher, J., Krishna, R., Mitra, R. et al.
De novo design of all-atom biomolecular interactions with RFdiffusion3.
bioRxiv (2025). https://doi.org/10.1101/2025.09.18.676967
[Preprint - citation will update on peer-reviewed publication.]

Ahern, W., Yim, J., Tischer, D. et al.
Atom-level enzyme active site scaffolding using RFdiffusion2.
Nature Methods 23, 96-105 (2025). https://doi.org/10.1038/s41592-025-02975-x
```

---

## Contact

For questions, bug reports, or collaboration inquiries:
- **Seth Woodbury** &mdash; woodbuse@uw.edu
- **Donghyo Kim** &mdash; donghyo@uw.edu
