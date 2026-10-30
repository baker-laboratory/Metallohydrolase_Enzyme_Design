# Contributing

[← Documentation](README.md)

Read the relevant directory README and inspect `git status --short` before
editing. [AGENTS.md](../AGENTS.md) gives guidance for coding agents.

## Preserve provenance

Keep raw measurements, deposited sequences/models, theozymes, and historical
outputs intact unless changing them is part of the task. For analysis changes,
explain the affected datasets, calibration or fitting choices, and changed
results. Distinguish fresh calculations from historical figure reproductions.

Notebook runs can overwrite tracked figures. Review those diffs separately
from documentation changes. Keep submodule pins stable during unrelated work.

## Keep setup portable

Use repository-relative paths and the existing
[environment](../Scripts/env_config.py) and [ligand-tool](../Scripts/repo_paths.py)
helpers. Keep checkpoints, credentials, local environments, and container images
out of Git. Put installation commands in the [environment guide](../Environment/README.md).

Keep analysis dependencies aligned across `analysis.yml`,
`requirements-analysis.txt`, and `pixi.toml`. Preserve `zinc_hydro.lock.yml` as
the historical export; document the platform and validation of any new solve.

## Check and describe changes

- Check links, filenames, commands, and citations; run `git diff --check`.
- Run focused checks for changed code or dependencies. Some existing tests
  under [Scripts](../Scripts/) require model packages or PyRosetta.
- Do not execute whole campaign notebooks to check prose: cells can submit jobs
  or generate large outputs.
- Describe what changed, why, the validation performed, and remaining limits.

Use [GitHub Issues](https://github.com/baker-laboratory/Metallohydrolase_Enzyme_Design/issues)
for problems; include the command, traceback, commit, and environment. Keep
bibliographic updates consistent across the README, [citation guide](citations.md),
[BibTeX](references.bib), and [CITATION.cff](../CITATION.cff).

## README colors

The [badge SVGs](assets/badges/) use IPD blue `#4D97B7`, pink `#B11186`, and
purple `#4B2E83`, matching the [official IPD logo](https://www.ipd.uw.edu/wp-content/uploads/2025/03/IPD-logo-UW-color-2025.png)
and the [named PyMOL colors](https://github.com/SethWoodbury/pymolrc/blob/4cfc4d9f1838696b34d0f430394fb2fda850654a/.pymolrc#L118-L124).
Keep dark text on blue and white text on pink/purple for legibility.
