# SIGMMA, σ
Spatial transcriptome–histology Image representation learning via hierarchical Graph-based Multi-scale Multi-modal Alignment

## Installation

SIGMMA is published on PyPI as [`sigmma`](https://pypi.org/project/sigmma/).
It targets **Linux x86_64, Python 3.11, PyTorch 2.3.1, torchvision 0.18.1,
NumPy 1.x and DGL 2.4.0**.
Choose either CPU or CUDA 12.1; install the matching PyTorch and DGL builds
before installing `sigmma`. DGL is a required runtime dependency.
The package declares `dgl==2.4.0`, which accepts both `2.4.0` and `2.4.0+cu121`
builds. DGL 2.4.0 is not available on PyPI, so `pip install sigmma` alone
cannot resolve it: install DGL from the official DGL index first, as below.

**Validation status (2026-11-12):** fresh environments using official DGL team
Conda binaries passed SIGMMA wheel installation, `pip check`, core imports,
graph forward/backward checks on CPU and A100 CUDA 12.1, and all four CLI
`--help` checks. See [the tested Conda route](docs/installation.md#validated-conda-route)
and `packaging_outputs/conda_validation_20261112/` for commands and evidence.
This validates the locally built SIGMMA wheel, not the PyPI artifact or model
accuracy. Direct DGL pip wheel installation remains unverified: `data.dgl.ai`
still terminates TLS before sending a certificate. Previous GPU gradient
comparisons used a diagnostic objective containing KL; they do not establish
training-gradient equivalence.

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

Then install SIGMMA from PyPI:

```bash
python -m pip install sigmma
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
does not validate GPU execution. Use a regular `pip install sigmma` for this
check, not an editable install that could hide missing package files.

If `data.dgl.ai` fails during TLS negotiation, stop at the DGL installation step.
Do not silently substitute another DGL version or disable TLS verification.
Use the separate wheel download procedure below if the installation machine
cannot access the official DGL index.
The historical research `poetry.lock` and `uv.lock` are not release locks for
this release; fresh release locks await access to the stable DGL artifacts.

### Download and install the DGL wheel separately

You can download DGL separately and transfer the wheel to the installation
machine. Use the official index matching your PyTorch build:

- **CPU:** [DGL wheels for PyTorch 2.3](https://data.dgl.ai/wheels/torch-2.3/repo.html).
  Select DGL **2.4.0**, Python **3.11** (`cp311`), Linux **x86_64**.
- **CUDA 12.1:** [DGL CUDA 12.1 wheels for PyTorch 2.3](https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html).
  Select DGL **2.4.0+cu121**, Python **3.11** (`cp311`), Linux **x86_64**.

Open the appropriate index in a browser and download the matching `.whl`, or
run one of the following commands on a machine that can reach `data.dgl.ai`.
These commands select the Linux target even when the download machine uses
macOS or Windows; they do not install DGL on the download machine.

```bash
# CPU wheel
python -m pip download --no-deps --only-binary=:all: \
  --platform manylinux2014_x86_64 --python-version 311 \
  --implementation cp --abi cp311 --no-index \
  --find-links https://data.dgl.ai/wheels/torch-2.3/repo.html \
  --dest dgl-wheel-cpu 'dgl==2.4.0'

# OR: CUDA 12.1 wheel
python -m pip download --no-deps --only-binary=:all: \
  --platform manylinux2014_x86_64 --python-version 311 \
  --implementation cp --abi cp311 --no-index \
  --find-links https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html \
  --dest dgl-wheel-cu121 'dgl==2.4.0+cu121'
```

Transfer the selected `.whl` using SFTP or `scp`. In the new environment on the
installation machine, install NumPy and the matching PyTorch/torchvision build
using the CPU or CUDA instructions above, then replace the DGL index installation
command with a local wheel installation:

```bash
# Replace this path with the exact transferred DGL wheel filename.
python -m pip install /path/to/dgl-wheel.whl
python -m pip install sigmma
python -m pip check
python scripts/check_install.py --device cpu
# Use --device cuda instead for the CUDA environment on a GPU host.
```

The installation check script is supplied in the source checkout. Only the DGL
wheel is transferred by this procedure; other dependencies still require network
access or a separately prepared offline wheel collection. If the download
machine also encounters a TLS error, it cannot obtain the wheel through this
route either. Installing a transferred wheel does not validate online access to
the DGL index.

Sources: [DGL installation](https://www.dgl.ai/pages/start.html),
[DGL 2.4 release notes](https://github.com/dmlc/dgl/releases/tag/v2.4.0),
[PyTorch version-specific installation](https://pytorch.org/get-started/previous-versions/).

SIGMMA constructs UNI2-h with `timm`; no local UNI package is required.
Download the pretrained weights separately as described below.

TERRA (formerly NicheJEPA) is not a SIGMMA runtime dependency. SIGMMA reads
precomputed cell embeddings from `AnnData.obsm`; generating those embeddings
belongs to a separate preprocessing environment. Set `latent_embedding_name`
to the stored embedding key. The default is `terra` / `terra_latent`. If that
key is absent, automatic selection falls back to `nemo_latent` with a warning.
Explicit `latent_embedding_name` values are always used as written, without
fallback. Legacy `type_pretrained_cell_embedding: nemo` configs still select
`nemo_latent`. No embedding model is imported and no stored arrays are renamed.

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

## Comparison baselines

Comparison models are maintained in the separately installed
[`benchmarks/`](https://github.com/Lotfollahi-lab/SIGMMA2.0/blob/main/benchmarks/README.md) project and are excluded from the SIGMMA
wheel. Install `./benchmarks[retrieval]` to run retrieval baselines; the original
baseline script paths remain forwarding entrypoints. SIGMMA itself does not
require the benchmark package.

### Optional analysis and source installation

After installing the matching PyTorch and DGL builds above, install the core
from this checkout with `python -m pip install .`, or use
`python -m pip install '.[analysis]'` for downstream analysis dependencies.
Baseline dependencies belong to the separately installed `benchmarks/` project.
No default research group or cluster-local package sources are required.
The SIGMMA command-line entrypoints are `sigmma-train`, `sigmma-retrieval`,
`sigmma-extract-embeddings`, and `sigmma-precompute-hier-cache`.

### Data paths

Cluster-specific data defaults can be overridden with `SIGMMA_CACHE_ROOT`,
`SIGMMA_IN_HOUSE_DATA_ROOT`, `SIGMMA_PUBLIC_DATA_ROOT`, `SIGMMA_NFS_DATA_ROOT`,
`SIGMMA_SAMPLE_METADATA_CSV`, and `SIGMMA_SECTION_PANEL_CSV`.

## Data Preparation
You can download dataset in the following [link](https://huggingface.co/datasets/AnonymousAuthor42/preprocessed_hest1k_LUAD)

## Model Training
```
torchrun --standalone --nproc_per_node 1 -m sigmma.cli.train -config_file ${run_dir}/configs/hest1k-LUAD_train_sigmma.yaml -out_dir ${output_dir}/hest1k-LUAD_sigmma
```

Before training, follow [Required UNI2-h weights](#required-uni2-h-weights) to download the backbone checkpoint and configure its directory.

### Optional: hierarchical graph cache
By default the dataset builds each tile's hierarchical sub-FOV graph on the fly,
so no cache is needed. For large training runs where data loading is the
bottleneck, pre-compute the graphs once per manifest:

```
sigmma-precompute-hier-cache --manifest /path/to/MANIFEST_train.csv --out_dir /path/to/hier_cache \
  --fov_size 256 --num_resolutions 3 --tissue_threshold 20000 --n_workers 16
```

Then set `hier_cache_dir: /path/to/hier_cache` under each dataset split that
should use it. `--fov_size`, `--num_resolutions` and `--tissue_threshold` must
match the dataset's `fov_size`, `num_resolutions` and
`tissue_filtering_threshold`; a mismatching cache raises an error. Sections
without a cache file fall back to on-the-fly construction. Add
`--cache_format source-v2` (new empty `--out_dir` required) to key caches by
source paths and options instead of section name; rebuild such a cache if the
data is moved.

## Downstream Tasks
``` 
# 1. gene expression prediction task
python scripts/benchmark_gex_prediction.py -config_file ${run_dir}/configs/hest1k-LUAD_linear_probing_sigmma.yaml -out_dir ${output_dir}/outputs/hest1k-LUAD_linear_probing_gex_prediction_sigmma

# 2. cross-modal retrieval
sigmma-retrieval -config_file ${run_dir}/configs/hest1k-LUAD_retrieval_sigmma.yaml -out_dir ${output_dir}/hest1k-LUAD_retrieval_sigmma
```

Note. SIGMMA checkpoint from *Model training* step should be provided to checkpoint_path in the configurations.

### Cell latent embeddings
Set `latent_embedding_name` under each dataset split (`dataset.train`,
`dataset.validation`, or the evaluation dataset block) to the exact AnnData
`obsm` key, for example `latent_embedding_name: "terra_latent"` or
`latent_embedding_name: "cell_emb"`. The selected embedding is used for both
full-FOV and hierarchical graph features. Its width must match the model's
`spatial_encoder_input_dim`. Explicit keys override `type_pretrained_cell_embedding`;
older configs without the key retain the legacy type-to-key mapping.
Graph caches must match the dataset's cell ordering, coordinates and connectivity.
