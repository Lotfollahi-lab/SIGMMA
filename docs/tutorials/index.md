# Tutorials

Start with the quickstart, then choose a notebook for your task. Read the
{doc}`../user_guide` for input formats and checkpoint compatibility.

| Tutorial | What you will produce |
|---|---|
| {doc}`new_data_preprocessing` | Registered Xenium + H&E, TERRA features and a validated SIGMMA manifest |
| {doc}`quickstart` | A manifest, a runnable configuration, and paired FOV embeddings |
| {doc}`multiscale_embeddings` | Cell embeddings and spatial domains at three scales |
| {doc}`cross_modal_retrieval` | Paired recall and cross-section image-to-spatial retrieval |
| {doc}`cell_niche_transfer` | Cell type/niche predictions on a held-out image region |

## Data and checkpoints

The recorded notebooks use preprocessed HEST-1k Xenium sections: lung
adenocarcinoma `TENX118` and `TENX141`, and pancreatic adenocarcinoma `TENX116`.
These are examples, not files installed by `pip`. Supply local copies of the
aligned image, spatial data and indices, a trained sigmma checkpoint, and
UNI2-h weights. Cell niche transfer also requires cell boundaries and labels.
The public sigmma checkpoint download URL has not yet been specified here;
obtain the checkpoint together with its architecture configuration from the
model provider. See {doc}`../installation` for UNI2-h access.

Edit the **Paths** cell in each notebook. Relative defaults refer to the
notebook's working directory. These examples use legacy `nemo_latent` input
features; use the representation your checkpoint was trained with. The
multi-scale clustering notebook additionally needs `igraph`, `leidenalg` and
`plotly` installed in its runtime environment.

The notebooks retain outputs from earlier GPU runs. Path configuration has
been made portable; the notebooks have not been re-executed for this docs
update. Recorded plots illustrate the original data/checkpoint combination.

```{toctree}
:maxdepth: 1
:caption: Getting started

quickstart
new_data_preprocessing
```

```{toctree}
:maxdepth: 1
:caption: Embeddings and downstream tasks

multiscale_embeddings
cross_modal_retrieval
```

```{toctree}
:maxdepth: 1
:caption: Applications

cell_niche_transfer
```
