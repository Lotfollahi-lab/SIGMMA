# Tutorials

The tutorials use public HEST-1k Xenium sections and a trained SIGMMA checkpoint:
two lung adenocarcinoma (LUAD) sections (`TENX118`, `TENX141`) for the embedding
and retrieval tutorials, and one pancreatic adenocarcinoma (PAAD) section
(`TENX116`) for cell niche transfer. Each notebook was executed on a GPU node; outputs are
stored in the notebook. Set `DATA_DIR` and `CKPT` in the first cell to your copies.

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
