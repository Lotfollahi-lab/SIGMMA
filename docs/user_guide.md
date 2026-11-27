# User Guide

## Overview

SIGMMA learns a shared representation of H&E histology and spatial
transcriptomics. Each **field of view (FOV)** — a 256 × 256 pixel tile of the
H&E image — is paired with the cells that fall inside it. The image side is
encoded by the pathology foundation model **UNI2-h**, adapted with LoRA; the
transcriptomic side is encoded by a graph network running over a
**hierarchical cell graph**, whose nodes carry pretrained cell embeddings.
The two encoders are trained contrastively, so that an image tile and the
molecular profile of the same tissue region land close together.

Alignment happens at **three spatial scales** rather than one. The cell graph
is coarsened step by step, and the image tile is pooled to match, which yields
a *micro* (cell neighbourhood), a *meso* (micro-environment) and a *macro*
(niche) representation of every FOV and of every cell in it. The resulting
embeddings support cross-modal retrieval, multi-scale niche discovery and the
transfer of cell-type and niche annotations to slides with H&E only.

## The inference pipeline

Inference takes one preprocessed section and a trained checkpoint, and has
three stages:

1. **Configure** — describe the model architecture and point to the weights.
   The configuration must be the one the checkpoint was trained with.
2. **Load** — build the model with `load_model` and wrap the section in a
   `SpatialPatchDataset`, which tiles the image and builds a cell graph per FOV.
3. **Embed** — run the model over the FOVs to obtain paired image and spatial
   embeddings at each scale.

```python
import torch
import yaml
from ml_collections.config_dict import ConfigDict
from torch.utils.data import DataLoader, SequentialSampler

from sigmma.datasets import SpatialPatchDataset, multimodal_collate
from sigmma.utils import load_model

device = torch.device("cuda")

# 1. Configure: architecture + paths to the SIGMMA checkpoint and UNI2-h.
config = ConfigDict(yaml.safe_load(open("quickstart.yaml")))

# 2. Load the model and one aligned section.
model = load_model(
    config,
    pretrained_image_encoder=config.model.pretrained_image_encoder,
    pretrained_ckpt_path=config.model.checkpoint_path,
    train=False,
).to(device).eval()

dataset = SpatialPatchDataset(
    image_file="data/TENX118/hematoxylin_eosin.zarr",
    spatial_file="data/TENX118/spatial_data_processed.h5",
    indices_file="data/TENX118/index_mask_HAE_stride_256_tile_256.zarr",
    num_resolutions=3,
    latent_embedding_name="nemo_latent",   # the obsm key your checkpoint expects
)
loader = DataLoader(dataset, batch_size=64, sampler=SequentialSampler(dataset),
                    collate_fn=multimodal_collate)

# 3. Embed: one list entry per scale (micro, meso, macro).
image_emb, spatial_emb = [], []
with torch.no_grad(), torch.amp.autocast("cuda"):
    for batch in loader:
        out = model(batch["image"].to(device), batch["spatial_graph"].to(device))
        image_emb.append([e.float().cpu() for e in out["image_embedding"]])
        spatial_emb.append([e.float().cpu() for e in out["spatial_embedding"]])
```

The {download}`example configuration <_static/examples/quickstart.yaml>`
reproduces the architecture used throughout the tutorials. See the
{doc}`tutorials/index` for complete, step-by-step versions of this workflow and
the {doc}`api/index` for the full reference.

### Running from the command line

The same pipeline is available as a command. Sections are listed in a
**manifest** CSV, which the configuration points to through
`dataset.query_dataset.manifest_path`:

```bash
sigmma-extract-embeddings -config_file quickstart.yaml -out_dir outputs/embeddings
```

This writes `embeddings.npz` holding `image_emb_s{0,1,2}` and
`spatial_emb_s{0,1,2}`, each of shape `(n_fovs, 512)` with rows paired across
the two modalities. The {doc}`tutorials/quickstart` builds the manifest and the
configuration from scratch.

:::{note}
The saved embeddings are raw model outputs. L2-normalize them before computing
cosine similarities, and note that the file stores no tile coordinates or cell
IDs — use the Python workflow above, or the
{doc}`tutorials/cross_modal_retrieval` notebook, when you need them.
:::

## Three spatial scales

Every forward pass returns one embedding per scale, indexed from fine to coarse:

| Index | Scale | Captures |
| --- | --- | --- |
| `0` | micro | a cell and its immediate neighbours |
| `1` | meso | the local micro-environment |
| `2` | macro | the tissue niche across the whole FOV |

The pipeline above produces **FOV-level** embeddings: one pooled vector per
tile, per modality, per scale. The spatial encoder also yields **cell-level**
embeddings — the node features of the graph at each scale — which are what you
cluster to find spatial domains. The {doc}`tutorials/multiscale_embeddings`
notebook extracts these with `extract_graphs` and stores them in AnnData.

## Input data

SIGMMA starts from a section whose H&E image and cell coordinates are already
registered to each other. A section consists of three files:

| File | Contents |
| --- | --- |
| H&E image | Zarr array, RGB, channel-first `(3, height, width)` at 0.5 µm/pixel |
| Spatial data | AnnData (`.h5ad`) with `obsm["spatial"]` in micrometres, `obsp["spatial_connectivities"]`, and a cell embedding in `obsm` |
| Tile index | Zarr array `(n_tiles, 3)`: `y_start`, `x_start` in pixels, and a tissue score |

Tiles whose tissue score falls below `tissue_filtering_threshold` are skipped,
so set it to `0` when every cell must be embedded.

For multiple sections, list them in a manifest with the columns `sample_name`,
`hematoxylin_eosin_image`, `spatial_preprocessed`, `annotation` (optional) and
`index`. Give each section a distinct `sample_name`.

:::{note}
The command-line tools assume a 0.5 µm/pixel image. For any other resolution,
construct `SpatialPatchDataset` yourself and pass
`image_resolution_um_per_pixel`.
:::

### Choosing the cell representation

The node features of the cell graph are read from `adata.obsm`; SIGMMA does not
compute them. Set `latent_embedding_name` to the key your checkpoint was
trained on — `nemo_latent` for the tutorial checkpoint, `terra_latent` for
models trained on [TERRA](https://terra-st.readthedocs.io) embeddings. Two
representations of the same dimensionality are not interchangeable, so always
set this key explicitly rather than relying on the automatic fallback.

## Model weights

Two sets of weights are needed, and neither ships with the PyPI package:

| Weights | Configuration key | Points to |
| --- | --- | --- |
| UNI2-h backbone | `model.pretrained_image_encoder_ckpt_path` | the **directory** containing `pytorch_model.bin` |
| SIGMMA checkpoint | `model.checkpoint_path` | the checkpoint **file** |

UNI2-h is gated: accept its terms on the model page linked from
{doc}`installation`, then download it. A SIGMMA checkpoint comes either from
your own training run or from the model provider, together with its
configuration.

A checkpoint is only loadable with the architecture it was trained with — the
input and projection dimensions, the pooling head and the LoRA settings all
have to match. If loading fails with a shape mismatch, fix the configuration
rather than the checkpoint path.

## Cross-modal retrieval

Because both modalities share one embedding space, an H&E tile can retrieve the
spatial-transcriptomics regions that resemble it, and vice versa.
`sigmma-retrieval` scores this on a section against itself:

```bash
WANDB_MODE=disabled sigmma-retrieval \
  -config_file quickstart.yaml -out_dir outputs/retrieval
```

It writes the similarity matrices (`score_matrices.npz`) and recall metrics
(`recall_metrics.json`) at the scale selected by `test.scale_idx`. This is a
*paired* benchmark: each image is expected to retrieve its own region. To
search a different section — the realistic use case — follow the
{doc}`tutorials/cross_modal_retrieval` notebook, which builds a separate
reference bank.

## Speeding up data loading

By default each FOV's hierarchical graph is built on the fly. For repeated runs
over the same sections, precompute the graphs once with
`sigmma-precompute-hier-cache` and point `hier_cache_dir` at the result. Caches
are keyed by source path and options, so rebuild them after moving or changing
the input files.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| DGL cannot be installed with pip | Use the Conda route in {doc}`installation` |
| `CXXABI` error on import | Put the Conda environment's `lib` directory first on `LD_LIBRARY_PATH` |
| Missing `obsm` key | Inspect `list(adata.obsm)` and set `latent_embedding_name` |
| Checkpoint shape mismatch | Use the checkpoint's original configuration |
| No tiles retained | Lower `tissue_filtering_threshold`; check image/coordinate alignment |
| CUDA out of memory | Reduce the batch size |
