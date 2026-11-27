# SIGMMA, σ

**S**patial transcriptome–histology **I**mage representation learning via hierarchical
**G**raph-based **M**ulti-scale **M**ulti-modal **A**lignment.

SIGMMA aligns H&E image tiles with spatial transcriptomics at three spatial scales
(cell neighbourhood, micro-environment, niche) using a hierarchical cell graph on the
transcriptomic side and a pathology foundation model (UNI2-h) on the image side. The
shared embedding space supports cross-modal retrieval, gene expression prediction from
H&E, and transfer of cell niche annotations to new slides.

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} {octicon}`desktop-download;1.5em;sd-mr-1` Installation
:link: installation
:link-type: doc

Set up sigmma with the matching PyTorch and DGL builds, and download the UNI2-h weights.
:::

:::{grid-item-card} {octicon}`light-bulb;1.5em;sd-mr-1` User Guide
:link: user_guide
:link-type: doc

Understand the concepts, the inference pipeline, the input data and the three spatial scales.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Quickstart
:link: tutorials/quickstart
:link-type: doc

Run one section end to end and extract paired image and spatial embeddings.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Tutorials
:link: tutorials/index
:link-type: doc

Learn from worked notebooks: multi-scale embeddings, cross-modal retrieval and niche transfer.
:::

:::{grid-item-card} {octicon}`code-square;1.5em;sd-mr-1` API
:link: api/index
:link-type: doc

Detailed descriptions of sigmma's public functions and classes.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

installation
user_guide
tutorials/index
api/index
```
