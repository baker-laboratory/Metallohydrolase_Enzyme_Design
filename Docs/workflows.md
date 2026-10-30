# From active site to candidate enzyme

[← Documentation](README.md)

A **theozyme** defines catalytic residues, metal ions, and substrate or
transition-state geometry. Diffusion builds a scaffold around it; downstream
design and prediction select candidates for experimental testing.

## Choose a workflow

| Entry point | Included | Endpoint |
| :--- | :--- | :--- |
| [RFdiffusion2 tutorial](../RFdiffusion2_Tutorial/) | PDB/QM theozymes, ORI placement, YAML inputs, inference commands, filtering | Filtered scaffolds |
| [RFdiffusion3 tutorial](../RFdiffusion3_Tutorial/) | Both theozyme examples, JSON inputs, inference, filtering, Cartesian relaxation | Geometry-idealized scaffolds; supplied inference configs use 1QJI |
| [Metalloesterase campaign](../Design_Pipelines/Metalloesterase_RFdiffusion2/) | Driver, inputs, selected outputs | RFjoint2, ProteinMPNN, AF2, FastMPNNDesign, and PLACER assessment |
| [Phosphotriesterase campaign](../Design_Pipelines/Phosphotriesterase_RFdiffusion3/) | Driver and two-zinc/KCX inputs | LigandMPNN, apo/holo AF3, Foldseek, redesign, and apo AF2; production outputs not deposited |
| [Metalloprotease campaign](../Design_Pipelines/Metalloprotease_RFdiffusion3/) | Scope README and directory placeholders | Driver and inputs not deposited; experimental data available separately |

## Key decisions

1. **Prepare the motif.** Start with the supplied 1QJI or ZETA_2 example. Check
   residue identifiers, ligand atom names, catalytic `REMARK 666` records, and
   ORI positions. Remove ORI pseudo-atoms from scoring references as instructed.
2. **Generate scaffolds.** RFdiffusion2 uses YAML; RFdiffusion3 uses JSON and
   `rfd3 design`. Inspect one command before expanding a sweep.
3. **Filter and inspect.** Examine structures and metric distributions before
   choosing cutoffs. PyRosetta scoring requires ligand `.params` and suitable
   reference coordinates.
4. **Design and validate sequences.** Follow the chosen campaign's stages.
   Use publication sequence tables and manifests to identify tested constructs;
   intermediate model filenames may precede final sequence changes.

The [experimental-data guide](data-and-reproduction.md) connects model outputs
to measured activity.

## Run on your system

Install the [helper environment](../Environment/README.md) and separate
[model runtimes and weights](software-and-models.md). Notebook driver cells
write commands for terminal or scheduler execution; running every cell does
not execute and validate all external jobs.

Regenerate configs for your checkout and review launchers, output paths,
container mounts, and scheduler settings. The phosphotriesterase notebook
retains container commands, `/net` and `/home` bind examples, and alternative
AF3 launchers. An empty container variable does not convert those commands to
host Python. Its `_OUTPUT_DIR_OVERRIDE` setting supports scratch storage.

Keep bundled inputs and reference outputs intact. Record commits, checkpoint
identifiers, settings, and seeds: stochastic sampling need not recreate the
exact deposited candidates. See the [repository map](repository-map.md) for
shared scripts and [contributing](contributing.md) for change guidance.
