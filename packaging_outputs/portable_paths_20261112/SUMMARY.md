# Portable release validation

Status: passed (2026-11-12).

Server-specific defaults and path examples were removed from package source.
Mapping CSVs now require user configuration at call time. Cache defaults to
`~/.cache/sigmma` and accepts `SIGMMA_CACHE_ROOT` or an explicit `root`.
Transcript lookup uses manifest paths; legacy unconfigured roots are `None`.
An absent SVG mapping CSV now raises an error instead of silently changing labels.

Both wheel and sdist were scanned in full (48 files each), including metadata
and README. No server path markers were found. Python string literals starting
with `/` were also inspected; remaining values are separators, HDF5 keys, or
output filename suffixes, not server paths. Author attribution was preserved.

The new wheel was installed in the previously validated independent CPU/CUDA
Conda environments. Both passed four path-configuration tests, pip check, core
imports and graph forward/backward checks, and all four CLI help checks.
This is not a full training or accuracy evaluation.

The verified artifacts are in `dist/portable/` and copied to `dist/`.
Previous top-level dist artifacts were preserved locally in `previous_dist/`;
older historical builds are not the current release. Hashes and commands are
in `result.json`. Validation logs are repository-local and not included in
wheel or sdist. The artifacts have not been published.
