# SIGMMA comparison benchmarks

This is a separately installable research package (`sigmma_benchmarks`). It is
kept in the same repository for reproducibility, but is not included in the
`sigmma` wheel. SIGMMA does not import this package.

## Installation

Use a dedicated Python 3.11 environment with the appropriate PyTorch 2.3 / DGL
build. From the repository root, install the library and benchmark package:

```bash
python -m pip install -e .
python -m pip install -e './benchmarks[retrieval]'
```

`retrieval` adds OpenCLIP and Transformers for CLIP, PLIP, and OmiCLIP. OmiCLIP
needs its pretrained checkpoint, but does not need the Loki Python package.
ST-Net, DeepPT, and HisToGene use the benchmark package's base dependencies.
Phoenix additionally needs its upstream `phoenix` implementation and checkpoint;
install it in its own compatible environment following the upstream project's
instructions. It is not bundled or automatically installed here.

For an already provisioned research environment, use `--no-deps` on the editable
installs to preserve its versions; this does not validate those dependencies.
Keep conflicting methods in separate environments, exporting predictions or
embeddings for common evaluation. This project is deliberately not a member of
a shared uv workspace. Record the environment, upstream revision, configuration,
checkpoint and SIGMMA revision with each experiment.

## Running comparisons

```bash
python -m sigmma_benchmarks.runners.train_baseline_stnet --help
python -m sigmma_benchmarks.runners.train_baseline_deeppt --help
python -m sigmma_benchmarks.runners.train_baseline_histogene --help
python -m sigmma_benchmarks.runners.predict_baseline_gex_2d --help
python -m sigmma_benchmarks.runners.extract_cell_niche_features_baselines --help
python -m sigmma_benchmarks.runners.benchmark_retrieval --help
```

The original paths under `scripts/` remain forwarding entrypoints for existing
job scripts. They require this benchmark package to be installed. The shared
`scripts/train/train_model.py` launcher imports the benchmark model factory only
when a non-SIGMMA model is selected. Existing configs and job scripts remain at
their original locations so their data and checkpoint paths do not change.

## Boundaries and checkpoints

- `baselines/`: comparison models, their preprocessing, regression trainer and
  inference adapters.
- `models.py`: lazy model construction for retrieval comparisons.
- `runners/`: experiment CLI implementations.
- SIGMMA provides shared data, metrics and training utilities. External models
  declare `spatial_input_key` and are passed through the `model_factory` hook;
  the library never imports benchmark implementations.

Model constructors, parameter names and state-dict checkpoint formats are
preserved. Existing state-dict checkpoints do not need conversion. Pickled
whole-model objects referring to the old Python module paths are not supported;
load those in the old checkout and export a state dict first.

When copying or adapting third-party implementations, retain their attribution
and licenses and record the upstream revision and any scientific changes.

## Verification

Run the CPU boundary and checkpoint tests in a provisioned environment:

```bash
PYTHONPATH=benchmarks/src python -m unittest discover -s benchmarks/tests -v
```

These tests cover core/benchmark import separation, graph/tensor/text input
selection (including wrapped models), external model construction and state-dict
restoration, and BLEEP loss behavior. They do not replace dataset-level or GPU
training evaluation.
