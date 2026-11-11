# Installation

The current release candidate targets **Linux x86_64, Python 3.11,
PyTorch 2.3.1, torchvision 0.18.1, NumPy 1.x and DGL 2.4.0**.
Choose either CPU or CUDA 12.1; install the matching PyTorch and DGL builds
before installing the SIGMMA wheel. DGL is a required runtime dependency.
The wheel declares `dgl==2.4.0`, without a development-wheel URL or a local path.
This requirement accepts both `2.4.0` and `2.4.0+cu121` builds.

**Validation status:** this is a release candidate, not an installation-tested
support matrix. Access to `data.dgl.ai` failed with a TLS EOF in the development
environment. Complete installation and SIGMMA graph execution with stable DGL
2.4.0 remain unverified. Historical research used DGL `2.5a241214+cu121`;
those results do not validate the stable release candidate.

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

Cluster-specific default paths can be overridden with environment variables:
`SIGMMA_CACHE_ROOT`, `SIGMMA_IN_HOUSE_DATA_ROOT`, `SIGMMA_PUBLIC_DATA_ROOT`,
`SIGMMA_NFS_DATA_ROOT`, `SIGMMA_SAMPLE_METADATA_CSV`, `SIGMMA_SECTION_PANEL_CSV`.

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
