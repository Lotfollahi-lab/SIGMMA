"""Model construction for comparison experiments, outside SIGMMA's wheel."""
from pathlib import Path


def build_model(config, train=True):
    if config.model.name == "openAI_CLIP":
        from .baselines.openai_clip import openAI_CLIP
        model = openAI_CLIP(
            hf_model_id=config.model.get("hf_model_id", "openai/clip-vit-base-patch32"),
            freeze_encoders=not bool(train),
            loss_temperature=(float(config.model.contrastive_temperature)
                              if train and config.model.get("contrastive_temperature") is not None
                              else None),
            image_backbone=str(config.model.get("image_backbone", "default")),
            uni_ckpt_path=config.model.get("uni_ckpt_path", None),
        )

    elif config.model.name == "PLIP":
        from .baselines.plip import PLIP
        model = PLIP(
            hf_model_id=config.model.get("hf_model_id", "vinid/plip"),
            freeze_encoders=not bool(train),
            loss_temperature=(float(config.model.contrastive_temperature)
                              if train and config.model.get("contrastive_temperature") is not None
                              else None),
            image_backbone=str(config.model.get("image_backbone", "default")),
            uni_ckpt_path=config.model.get("uni_ckpt_path", None),
        )

    elif config.model.name == "loki_OmiCLIP":
        from .baselines.omiclip import loki_OmiCLIP
        ckpt = config.model.pretrained_ckpt_path
        assert Path(ckpt).exists(), f"OmiCLIP checkpoint not found at {ckpt}"
        model = loki_OmiCLIP(
            pretrained_ckpt_path=str(ckpt),
            freeze_encoders=not bool(train),
            loss_temperature=(float(config.model.contrastive_temperature)
                              if train and config.model.get("contrastive_temperature") is not None
                              else None),
            image_backbone=str(config.model.get("image_backbone", "default")),
            uni_ckpt_path=config.model.get("uni_ckpt_path", None),
        )

    elif config.model.name == "BLEEP":
        from .baselines.bleep import BLEEP
        model = BLEEP(
            spot_input_dim=int(config.model.spot_input_dim),
            image_embedding_dim=int(config.model.get("image_embedding_dim", 2048)),
            projection_dim=int(config.model.get("projection_dim", 256)),
            spot_hidden_dim=int(config.model.get("spot_hidden_dim", 512)),
            temperature=float(config.model.get("contrastive_temperature", 1.0)),
            dropout=float(config.model.get("dropout", 0.1)),
            image_size=int(config.model.get("image_size", 224)),
            pretrained_image_encoder=bool(config.model.get("pretrained_image_encoder_imagenet", True)),
            image_backbone=str(config.model.get("image_backbone", "default")),
            uni_ckpt_path=config.model.get("uni_ckpt_path", None),
        )

    else:
        raise NotImplementedError(f"Unknown benchmark model: {config.model.name}")
    return model


def load_model(config, pretrained_image_encoder="UNI", pretrained_ckpt_path=None,
               train=True, rank=None):
    """Load SIGMMA or a comparison model through the common device/checkpoint path."""
    from sigmma.utils.train_utils import load_model as load_core_model
    return load_core_model(
        config, pretrained_image_encoder=pretrained_image_encoder,
        pretrained_ckpt_path=pretrained_ckpt_path, train=train, rank=rank,
        model_factory=build_model,
    )
