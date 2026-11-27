# Quickstart: paired FOV embeddings

Run one preprocessed section through sigmma and inspect its image and spatial
embeddings. This walkthrough uses local assets that you supply. It does not
download a sigmma checkpoint or preprocess raw Xenium data.

## Before you start

Complete {doc}`../installation` and read the input contract in
{doc}`../user_guide`. You need a CUDA GPU, a trained sigmma checkpoint and its
matching configuration, approved UNI2-h weights, and one aligned section.

For example, organize your own files as follows:

```text
my-sigmma-work/
├── data/TENX118/
│   ├── hematoxylin_eosin.zarr/
│   ├── index_mask_HAE_stride_256_tile_256.zarr/
│   └── spatial_data_processed.h5
└── weights/
    ├── sigmma.pth
    └── UNI2-h/pytorch_model.bin
```

`TENX118` is an example identifier, not a dataset bundled with the package.
The existing notebooks also use `TENX141` for retrieval and `TENX116` for
annotation transfer. You may substitute your own compatible section.

## 1. Configure paths and check the inputs

Run the following in a notebook or Python session from your working directory.
Change `ROOT`, the filenames and `LATENT_KEY` to match your data. The illustrated
checkpoint expects 384-dimensional legacy `nemo_latent` cell features.

```python
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import zarr

ROOT = Path(".").resolve()
SECTION = "TENX118"
LATENT_KEY = "nemo_latent"
INPUT_DIM = 384  # must match the trained checkpoint
section = ROOT / "data" / SECTION
image_file = section / "hematoxylin_eosin.zarr"
index_file = section / "index_mask_HAE_stride_256_tile_256.zarr"
spatial_file = section / "spatial_data_processed.h5"
checkpoint = ROOT / "weights" / "sigmma.pth"
uni_dir = ROOT / "weights" / "UNI2-h"

for path in (image_file, index_file, spatial_file, checkpoint,
             uni_dir / "pytorch_model.bin"):
    if not path.exists():
        raise FileNotFoundError(f"Supply this asset first: {path}")
assert torch.cuda.is_available(), "Use a CUDA environment for this tutorial."
adata = ad.read_h5ad(spatial_file)
image = zarr.open(str(image_file), mode="r")
indices = zarr.open(str(index_file), mode="r")
assert len(image.shape) == 3 and image.shape[0] == 3
assert len(indices.shape) == 2 and indices.shape[1] == 3
assert indices.shape[0] > 0, "The tile index must not be empty."
assert adata.obsm["spatial"].shape == (adata.n_obs, 2)
assert np.isfinite(adata.obsm["spatial"]).all()
assert LATENT_KEY in adata.obsm, list(adata.obsm)
assert adata.obsm[LATENT_KEY].shape == (adata.n_obs, INPUT_DIM)
assert np.isfinite(adata.obsm[LATENT_KEY]).all()
assert sp.issparse(adata.obsp["spatial_connectivities"])
assert adata.obsp["spatial_connectivities"].shape == (adata.n_obs, adata.n_obs)
print(f"Inputs found: {adata.n_obs} cells; {indices.shape[0]} candidate tiles")
```

These checks verify structure, not registration quality. Confirm that the cell
coordinates are in micrometres and align to a 0.5 µm/pixel H&E image before
continuing.

## 2. Write a manifest and configuration

Download the {download}`example configuration <../_static/examples/quickstart.yaml>`
and save it as `quickstart.yaml` in `ROOT`. Its model section reproduces the
architecture in the worked notebooks; replace it with the configuration for
your checkpoint if different. In particular, do not change LoRA `per_scale`
just to suppress a checkpoint-loading error.

Continue in the same Python session:

```python
import yaml

manifest_path = ROOT / "manifest.csv"
pd.DataFrame([{
    "sample_name": SECTION,
    "hematoxylin_eosin_image": str(image_file),
    "spatial_preprocessed": str(spatial_file),
    "annotation": "",
    "index": str(index_file),
}]).to_csv(manifest_path, index=False)

config_path = ROOT / "quickstart.yaml"
config = yaml.safe_load(config_path.read_text())
config["model"]["checkpoint_path"] = str(checkpoint)
config["model"]["pretrained_image_encoder_ckpt_path"] = str(uni_dir)
config["dataset"]["query_dataset"]["manifest_path"] = str(manifest_path)
config["dataset"]["query_dataset"]["latent_embedding_name"] = LATENT_KEY
config_path.write_text(yaml.safe_dump(config, sort_keys=False))
```

The example starts with batch size 1 and no worker processes. Its tissue score
threshold is 20000, matching the default preprocessing convention. Inspect the
score column of your index before changing the threshold; tiles without cells
may be skipped even when their image passes the tissue filter.

## 3. Extract and inspect embeddings

From a terminal in `ROOT` with the same environment activated:

```bash
sigmma-extract-embeddings -config_file quickstart.yaml -out_dir outputs/embeddings
```

Then inspect the output in Python:

```python
features = np.load(ROOT / "outputs" / "embeddings" / "embeddings.npz")
for scale in range(3):
    image_features = features[f"image_emb_s{scale}"]
    spatial_features = features[f"spatial_emb_s{scale}"]
    assert image_features.shape == spatial_features.shape
    assert np.isfinite(image_features).all() and np.isfinite(spatial_features).all()
    print(scale, image_features.shape)
```

Expect six arrays: two modalities at each of three scales. Their first
dimension is the number of retained FOVs, not the number of cells. The example
projection dimension is 512. Coordinates are not saved by this CLI; the
{doc}`cross_modal_retrieval` notebook shows how to retain them.

## 4. Evaluate paired retrieval

```bash
WANDB_MODE=disabled sigmma-retrieval \
  -config_file quickstart.yaml -out_dir outputs/retrieval
```

```python
import json

metrics_path = ROOT / "outputs" / "retrieval" / "recall_metrics.json"
print(json.loads(metrics_path.read_text()))
```

This checks whether an image retrieves its own paired spatial region within
the selected section. It is not a cross-section generalization result.
Use at least 100 retained FOVs if interpreting recall at 1%, since the current
implementation rounds percentage cutoffs down.

## Next steps

- {doc}`multiscale_embeddings`: cell-level representations, clustering and spatial plots.
- {doc}`cross_modal_retrieval`: separate query/reference sections and retrieval visualizations.
- {doc}`cell_niche_transfer`: image features for annotation transfer to a held-out region.

The quickstart is a recipe for user-supplied data. Documentation builds do not
execute model inference or download weights, and no example metric is promised
for a different checkpoint or dataset.
