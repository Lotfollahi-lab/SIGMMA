# SIGMMA, σ

**S**patial transcriptome–histology **I**mage representation learning via hierarchical
**G**raph-based **M**ulti-scale **M**ulti-modal **A**lignment.

SIGMMA aligns H&E image tiles with spatial transcriptomics at three spatial scales
(cell neighbourhood, micro-environment, niche) using a hierarchical cell graph on the
transcriptomic side and a pathology foundation model (UNI2-h) on the image side. The
shared embedding space supports cross-modal retrieval, gene expression prediction from
H&E, and transfer of cell niche annotations to new slides.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Installation
:link: installation
:link-type: doc
Install SIGMMA and the DGL wheel, and separately download the UNI2-h checkpoint.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
End-to-end notebooks: embeddings, retrieval, gene expression and niche transfer.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc
Data loading, model loading, evaluation and application modules.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

installation
tutorials/index
api/index
```
