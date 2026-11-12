# Installation

The current release candidate targets **Linux x86_64, Python 3.11,
PyTorch 2.3.1, torchvision 0.18.1, NumPy 1.x and DGL 2.4.0**.
Choose either CPU or CUDA 12.1; install the matching PyTorch and DGL builds
before installing the SIGMMA wheel. DGL is a required runtime dependency.
The wheel declares `dgl==2.4.0`, without a development-wheel URL or a local path.
This requirement accepts both `2.4.0` and `2.4.0+cu121` builds.

**Validation status (2026-11-12):** the official DGL team Conda route below
passed fresh installation of the locally built SIGMMA wheel, `pip check`, core
imports, graph forward/backward checks on CPU and A100 CUDA 12.1, and all four
CLI `--help` checks. Direct installation of DGL pip wheels remains unverified
because `data.dgl.ai` terminates TLS before sending a certificate. These checks
do not validate the published SIGMMA PyPI artifact, model weights, training
equivalence, or biological evaluation results.

## Validated Conda route

Use micromamba (tested) or a compatible Conda installation. From the SIGMMA
source checkout, choose one profile:

```bash
# CPU
micromamba create -y -n sigmma-cpu \
  -c https://conda.anaconda.org/dglteam/label/th23_cpu -c conda-forge \
  python=3.11 pip dgl=2.4.0.th23.cpu 'numpy>=1.26,<2'
micromamba activate sigmma-cpu
python -m pip install 'torch==2.3.1' 'torchvision==0.18.1' --index-url https://download.pytorch.org/whl/cpu

# OR CUDA 12.1
micromamba create -y -n sigmma-cuda \
  -c https://conda.anaconda.org/dglteam/label/th23_cu121 -c conda-forge \
  python=3.11 pip dgl=2.4.0.th23.cu121 'numpy>=1.26,<2'
micromamba activate sigmma-cuda
python -m pip install 'torch==2.3.1' 'torchvision==0.18.1' --index-url https://download.pytorch.org/whl/cu121
```

Then, in the chosen environment:

```bash
# Prefer the environment's C++ runtime over an older system libstdc++.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DGLBACKEND=pytorch
python -m pip wheel --no-deps --wheel-dir ./dist/validated .
python -m pip install ./dist/validated/sigmma-2.0.0-py3-none-any.whl
python -m pip check
python scripts/check_install.py --device cpu  # use --device cuda for CUDA
sigmma-train --help
sigmma-retrieval --help
sigmma-extract-embeddings --help
sigmma-precompute-hier-cache --help
```

The explicit library path was necessary on the validation host: otherwise
SciPy loaded an older system `libstdc++.so.6` lacking `CXXABI_1.3.15`.
No model code or installed package metadata was modified. DGL's Python
distribution versions are `2.4.0` and `2.4.0+cu121`, satisfying SIGMMA's existing
dependency. Full inventories, Conda provenance, wheel hashes, initial failures,
and passing results are recorded in `packaging_outputs/conda_validation_20261112/`.

## Direct pip route (DGL host currently unavailable)

Start in a new environment:

```bash
python3.11 -m venv .venv-sigmma
source .venv-sigmma/bin/activate
python -m pip install --upgrade pip
```

For CPU:

```bash
python -m pip install 'numpy>=1.26,<2'
python -m pip install 'torch==2.3.1' 'torchvision==0.18.1' --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'dgl==2.4.0' --find-links https://data.dgl.ai/wheels/torch-2.3/repo.html
```

For CUDA 12.1 (with a compatible NVIDIA driver):

```bash
python -m pip install 'numpy>=1.26,<2'
python -m pip install 'torch==2.3.1' 'torchvision==0.18.1' --index-url https://download.pytorch.org/whl/cu121
python -m pip install 'dgl==2.4.0+cu121' --find-links https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
```

Then install the wheel supplied with the release (replace the example path):

```bash
python -m pip install /path/to/sigmma-2.0.0-py3-none-any.whl
python -m pip check
```

From a source checkout, run the installation check using that environment:

```bash
python scripts/check_install.py --device cpu
# Or, on a GPU host using the CUDA environment:
python scripts/check_install.py --device cuda
```

The check verifies versions, imports SIGMMA from the selected environment, and
runs small DGL forward/backward operations. It does not download UNI2-h weights
or establish checkpoint accuracy. CPU and CUDA checks are separate; a CPU pass
does not validate GPU execution. Use a regular wheel install for this check,
not an editable install that could hide missing wheel files.

If `data.dgl.ai` fails during TLS negotiation, stop at the DGL installation step.
Do not silently substitute another DGL version or disable TLS verification.
Obtaining the matching wheel on a connected machine and installing it locally
is an offline alternative; it does not establish online installation success.
The historical research `poetry.lock` and `uv.lock` are not release locks for
this candidate; fresh release locks await access to the stable DGL artifacts.

Sources: [DGL installation](https://www.dgl.ai/pages/start.html),
[DGL 2.4 release notes](https://github.com/dmlc/dgl/releases/tag/v2.4.0),
[PyTorch version-specific installation](https://pytorch.org/get-started/previous-versions/).

## Required UNI2-h weights

SIGMMA's default image backbone is **UNI2-h**. Its pretrained weights are an
external model asset: they are not included in the SIGMMA wheel and are not
automatically downloaded by package installation.

1. Request access and accept the terms on the official
   [MahmoodLab/UNI2-h model page](https://huggingface.co/MahmoodLab/UNI2-h).
   Use an approved Hugging Face account for the download.
2. On an internet-connected machine, use the
   [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
   to authenticate and download the checkpoint. With `uv` installed:

   ```bash
   uvx hf auth login
   uvx hf download MahmoodLab/UNI2-h pytorch_model.bin --local-dir /path/to/UNI2-h
   ```

3. If the machine running SIGMMA is offline, transfer the downloaded file to
   that machine. Set the **directory containing `pytorch_model.bin`**, not the
   file itself, in the SIGMMA configuration:

   ```yaml
   model:
     pretrained_image_encoder: UNI
     pretrained_image_encoder_ckpt_path: /path/to/UNI2-h
   ```

The configuration name `UNI` selects SIGMMA's internal wrapper, whose default
variant is UNI2-h. Use UNI2-h weights for this default, not the older UNI v1
checkpoint. The separate trained SIGMMA checkpoint belongs in
`model.checkpoint_path`. Download access and use remain subject to the upstream
model terms; SIGMMA does not redistribute the weights.

For the SIGMMA checkpoint used in the tutorials, see {doc}`tutorials/index`.

## Data paths

Specify data paths in your manifest and checkpoint paths in your model
configuration. No server-specific data locations are built into the package.

```bash
# Replace these examples with your own locations; set before starting Python.
export SIGMMA_CACHE_ROOT="$HOME/sigmma-cache"
export SIGMMA_SAMPLE_METADATA_CSV="$HOME/my-data/sample_metadata.csv"
export SIGMMA_SECTION_PANEL_CSV="$HOME/my-data/section_to_gene_panel.csv"
```

- Cache helpers default to `~/.cache/sigmma`; their `root=` argument overrides
  the default. `SIGMMA_CACHE_ROOT` is read at import time.
- `load_tissue_map(csv_path=...)` accepts a CSV with `sample_id,tissue_type`.
  Without an explicit argument it reads `SIGMMA_SAMPLE_METADATA_CSV`.
- `sigmma.evaluation.svg_selection.tissue_of(..., csv_path=...)` accepts a CSV
  with `sample_name,tissue`, or reads `SIGMMA_SECTION_PANEL_CSV`.
- Missing mapping settings/files raise an actionable error when those mapping
  functions are called; they do not prevent importing or training SIGMMA.
- Transcript binning resolves `transcripts.parquet` beside the H&E zarr path
  supplied in the manifest. No global data root is required.

Migration: set the variables above to your existing local paths to reuse your
data and caches. The legacy transcript-root constants remain available through
`SIGMMA_IN_HOUSE_DATA_ROOT`, `SIGMMA_PUBLIC_DATA_ROOT`, and `SIGMMA_NFS_DATA_ROOT`,
but are `None` when unset and are not used by transcript resolution.

## Development environment

The source checkout declares its Python dependencies in `pyproject.toml`.
No local UNI checkout is required: SIGMMA constructs the backbone with `timm`
and loads the separately downloaded UNI2-h weights. DGL must be available from
its external wheel host for installation to complete.

TERRA (formerly NicheJEPA) is not required to run SIGMMA. Generate cell embeddings
in a separate preprocessing environment and store them in `AnnData.obsm`.
The default embedding type is `terra`, selecting `terra_latent`. When this key
is absent, automatic selection uses legacy `nemo_latent` with a warning.
An explicit `latent_embedding_name` is used exactly as specified; a missing
explicit key raises an error rather than falling back. Legacy
`type_pretrained_cell_embedding: nemo` still selects `nemo_latent`.
Stored arrays are not renamed or converted.

## Comparison baselines

Baseline implementations live in the separate `benchmarks/` project, outside
the SIGMMA wheel. From a source checkout, install them with
`python -m pip install -e './benchmarks[retrieval]'` in a compatible environment.
See `benchmarks/README.md` for entrypoints and environment separation.
