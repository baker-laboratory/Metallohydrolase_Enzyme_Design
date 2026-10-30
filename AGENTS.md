# Repository guidance

This is a scientific design and data repository. Read `README.md` and the relevant directory README before changing a workflow; `Docs/README.md` indexes the detailed guides.

## Preserve scientific provenance

- Keep raw measurements, deposited sequences/models, theozymes, notebook outputs and manuscript figures unchanged unless the task explicitly calls for those changes. Do not regenerate scientific outputs as a side effect of documentation work.
- Treat untracked files as user work. Do not add, remove, or incorporate them into public documentation without task-specific authorization.
- Distinguish computational design models, experimental structures, fitted results and raw measurements. Do not imply that a generated design is experimentally validated.
- Preserve the commits pinned by `Software/` submodules. Avoid editing submodule contents or updating their revisions as incidental cleanup.

## Work within the existing structure

- Design notebooks live in `RFdiffusion*_Tutorial/` and `Design_Pipelines/`; experimental analysis entry points are `Manuscript_Data/*/wetlab_data_analysis.py` (marimo). Historical `.ipynb` analyses are retained separately.
- Shared implementation belongs in `Scripts/`; runtime and path configuration is handled by `Scripts/env_config.py`, `Scripts/repo_paths.py` and the notebook helpers. Use repository-relative paths and existing environment overrides rather than workstation paths.
- `Environment/analysis.yml` is the lightweight experimental-analysis environment. `Environment/zinc_hydro.yml` supplies design helpers; external model runtimes and weights require their own setup.
- Describe readiness from actual tracked content. The metalloprotease design directory is a placeholder; publication of its experimental results does not mean its campaign notebook is present.

## Validate the change

- For documentation, check relative links and referenced filenames, run `git diff --check`, and review the diff for unsupported execution claims or accidental data changes.
- For code, run the focused checks relevant to the change. Existing checks include `Scripts/tests/test_process_placer.py` and tests under `Scripts/mpnn_relevant_utils/`; some require model dependencies or weights. State what ran and what was unavailable.
- Do not launch production GPU/SLURM jobs or execute whole scientific notebooks merely to validate prose. Separate syntax/path checks from verified model execution and numerical reproduction in reports.
