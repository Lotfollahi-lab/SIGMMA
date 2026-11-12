# API reference

## Data and model loading

```{eval-rst}
.. currentmodule:: sigmma.utils.train_utils

.. autosummary::
   :toctree: generated

   load_dataset
   load_model
```

## Evaluation

```{eval-rst}
.. currentmodule:: sigmma.evaluation

.. autosummary::
   :toctree: generated

   evaluate_retrieval.eval_retrieval
   linear_probing.linear_probing

.. Written by hand: @torch.no_grad() hides the function from autodoc when
   torch is mocked on Read the Docs.

.. py:function:: sigmma.evaluation.evaluate_retrieval.retrieval_score(model, query_dataloader, reference_dataloader, device, config)
   :no-index:

   Embed the query and reference dataloaders with ``model`` and return the cosine
   similarity matrices ``(scores_img2spatial, scores_spatial2img)`` as numpy arrays.
   For SIGMMA, ``config.test.scale_idx`` selects the scale (``-1`` = macro, default).
```

## Applications: cell niche transfer

```{eval-rst}
.. currentmodule:: sigmma.application.cell_niche_transfer

.. autosummary::
   :toctree: generated

   process_slide_data
   process_slide_features
   encode_with_mask_pool
   rasterize_polygon
   extract_centered_crop
```
