# Repository map

[← Documentation](README.md)

The repository separates reusable design machinery from the experiments that evaluate it. The main directories are:

| Path | Contents | Start here when… |
|---|---|---|
| [RFdiffusion2_Tutorial/](../RFdiffusion2_Tutorial/) | Guided theozyme preparation, RFdiffusion2 scaffold generation and filtering | Learning the RFdiffusion2 workflow |
| [RFdiffusion3_Tutorial/](../RFdiffusion3_Tutorial/) | Guided theozyme preparation, RFdiffusion3 generation, filtering and geometry idealization | Learning the RFdiffusion3 workflow |
| [Design_Pipelines/](../Design_Pipelines/) | Campaign drivers and inputs; saved outputs where deposited | Inspecting campaign-specific choices or adapting a campaign |
| [Manuscript_Data/](../Manuscript_Data/) | Publication datasets, marimo analyses, sequence/model deposits and figures | Exploring experimental results or reproducing analyses |
| [Scripts/](../Scripts/) | Shared Python helpers and command-line drivers | Understanding or changing an implementation |
| [Software/](../Software/) | Pinned upstream Git submodules | Installing the external models and tools used by a workflow |
| [Environment/](../Environment/) | Analysis and design-helper environment specifications, container recipe and setup guidance | Preparing a machine to run selected stages |
| [Docs/](README.md) | Navigation, setup, workflow scope, data guidance and citations | Finding a route through the repository |

The [workflow guide](workflows.md) distinguishes available campaign drivers and
outputs from placeholders. The metalloprotease campaign has experimental data
but no deposited design driver.

## Scripts by responsibility

| Location | Responsibility |
|---|---|
| [env_config.py](../Scripts/env_config.py), [repo_paths.py](../Scripts/repo_paths.py) | Runtime and external-tool path resolution |
| [General_NoteBook_Functions.py](../Scripts/General_NoteBook_Functions.py), [notebook_functions/](../Scripts/notebook_functions/) | Notebook utilities, paths and batch-command support |
| [theozyme_and_ligand_handling/](../Scripts/theozyme_and_ligand_handling/) | Active-site preparation, ligand parameterization and motif sampling |
| [scaffold_handling/](../Scripts/scaffold_handling/), [design_filtering/](../Scripts/design_filtering/) | Scaffold idealization, geometry and pocket metrics |
| [mpnn_relevant_utils/](../Scripts/mpnn_relevant_utils/) | LigandMPNN orchestration, fixed residues, protonation and validation scripts |
| [enzyme_design/](../Scripts/enzyme_design/), [scoring/](../Scripts/scoring/) | Design/refinement helpers and scoring |
| [af2_analysis_and_tools/](../Scripts/af2_analysis_and_tools/), [advanced_structure_prediction_tools/](../Scripts/advanced_structure_prediction_tools/) | AF2/AF3 input and output handling |
| [wetlab_platereader/](../Scripts/wetlab_platereader/), [experimental_data_processing_functions.py](../Scripts/experimental_data_processing_functions.py) | Experimental analysis, kinetic fitting and plotting |
| [tests/](../Scripts/tests/) | Focused tests, including PLACER output processing |

Follow the script referenced by your notebook; similarly named historical entry points may differ.

## What is tracked, generated or external

- **Tracked scientific material:** example inputs, raw measurements, sequence/model archives, selected campaign intermediates and reference figures. These files support interpretation and reproduction; preserve their provenance.
- **Generated run material:** command lists, job scripts, configs, logs, scratch structures and new figures. Some examples are tracked, while larger runtime outputs are ignored. Check the path and Git status before regenerating them.
- **External resources:** model checkpoints, large databases, licensed software and local containers. A populated `Software/` checkout supplies source code, not all of these resources.

The submodule commits recorded by Git define the upstream code versions. The `production` branch hint for foundry in `.gitmodules` does not replace its pinned commit. See [software and models](software-and-models.md) for setup and [data and reproduction](data-and-reproduction.md) for the scientific deposits.
