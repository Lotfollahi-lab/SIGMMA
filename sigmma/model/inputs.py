"""Input selection shared by training and embedding evaluation."""

def spatial_input(model, batch, device):
    """Select spatial input using a model's optional spatial_input_key attribute.

    SIGMMA defaults to spatial_graph. External models may select a tensor or
    text field. This attribute is configuration, not a checkpoint parameter.
    """
    module = getattr(model, "module", model)
    key = getattr(module, "spatial_input_key", "spatial_graph")
    spatial = batch[key]
    return spatial.to(device) if hasattr(spatial, "to") else spatial
