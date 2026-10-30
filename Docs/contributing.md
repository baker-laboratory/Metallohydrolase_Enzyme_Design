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
