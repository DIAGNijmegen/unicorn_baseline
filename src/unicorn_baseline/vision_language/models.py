import json
import os

import timm
import torch
import torch.nn as nn
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked

from unicorn_baseline.vision.pathology.model_utils import update_state_dict


class Virchow(nn.Module):
    """
    Tile-level feature extractor.
    """

    def __init__(self, model_dir, input_size=224):
        super().__init__()
        self.model_dir = model_dir
        self.input_size = input_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model configuration
        with open(os.path.join(self.model_dir, "virchow-config.json"), "r") as f:
            self.config = json.load(f)

        if input_size == 256:
            self.config["pretrained_cfg"]["crop_pct"] = (
                224 / 256
            )  # Ensure Resize is 256

        # Initialize tile encoder
        self.tile_encoder = timm.create_model(
            **self.config, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU
        )

        self.load_weights()
        self.transforms = self.get_transforms()

    def load_weights(self):
        """Load pretrained weights for the tile encoder."""
        checkpoint_path = os.path.join(self.model_dir, "virchow-tile-encoder.pth")
        print(f"Loading tile encoder weights from {checkpoint_path}...")
        weights = torch.load(checkpoint_path, map_location=self.device)
        updated_sd, msg = update_state_dict(
            model_dict=self.tile_encoder.state_dict(), state_dict=weights
        )
        print(msg)
        self.tile_encoder.load_state_dict(updated_sd, strict=True)
        self.tile_encoder.to(self.device)
        self.tile_encoder.eval()

    def get_transforms(self):
        """Retrieve the transformation pipeline for input images."""
        data_config = resolve_data_config(
            self.config["pretrained_cfg"], model=self.tile_encoder
        )
        return create_transform(**data_config)

    def forward(self, x):
        """Extract tile-level embeddings."""
        x = x.to(self.device)
        with torch.no_grad():
            output = self.tile_encoder(x)

        # Extract class and patch tokens
        class_token = output[:, 0]
        patch_tokens = output[:, 1:]
        embedding = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)
        return embedding


class PRISM(nn.Module):
    """
    Slide-level feature extractor (PRISM model).
    """

    def __init__(self, model_dir, input_size=224):
        super().__init__()
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize encoder
        self.slide_encoder = self.build_slide_encoder()

    def build_slide_encoder(self):
        """Load slide encoder (PRISM)."""
        import sys

        sys.path.insert(0, self.model_dir)
        from unicorn_baseline.vision_language.prism.configuring_prism import (
            PerceiverConfig,
            PrismConfig,
        )
        from unicorn_baseline.vision_language.prism.modeling_prism import Prism
        from transformers.models.biogpt.configuration_biogpt import BioGptConfig

        cfg = PrismConfig(
            biogpt_config=BioGptConfig(),
            perceiver_config=PerceiverConfig(),
            model_dir=self.model_dir,
        )
        slide_encoder = Prism(cfg)

        checkpoint_path = os.path.join(self.model_dir, "prism-slide-encoder.pth")
        print(f"Loading slide encoder weights from {checkpoint_path}...")
        weights = torch.load(checkpoint_path, map_location=self.device)
        updated_sd, msg = update_state_dict(
            model_dict=slide_encoder.state_dict(), state_dict=weights
        )
        print(msg)
        slide_encoder.load_state_dict(updated_sd, strict=True)
        slide_encoder.to(self.device)
        slide_encoder.eval()

        return slide_encoder

    def forward_slide(self, embeddings):
        """Generate slide-level captions from tile embeddings."""
        embeddings = embeddings.to(self.device)

        reprs = self.slide_encoder.slide_representations(embeddings)

        genned_ids = self.slide_encoder.generate(
            key_value_states=reprs["image_latents"],
            do_sample=False,
            num_beams=5,
            num_beam_groups=1,
        )
        genned_caption = self.slide_encoder.untokenize(genned_ids)

        return genned_caption
