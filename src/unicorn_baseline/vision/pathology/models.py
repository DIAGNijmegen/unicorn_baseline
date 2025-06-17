from pathlib import Path
import os 
import json

import torch
import torch.nn as nn

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked

from unicorn_baseline.vision.pathology.model_utils import update_state_dict
from unicorn_baseline.vision.pathology.titan.configuration_titan import TitanConfig
from unicorn_baseline.vision.pathology.titan.modeling_titan import Titan


class SlideFeatureExtractor(nn.Module):
    def __init__(self, input_size: int = 224):
        self.input_size = input_size
        super(SlideFeatureExtractor, self).__init__()
        self.build_encoders()
        self.set_device()
        for param in self.parameters():
            param.requires_grad = False

    def set_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def build_encoders(self):
        raise NotImplementedError

    def get_transforms(self):
        return self.tile_encoder.get_transforms()

    def forward(self, x):
        return self.tile_encoder(x)

    def forward_slide(self, **kwargs):
        return self.slide_encoder(**kwargs)

    def __repr__(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"{self.__class__.__name__}\n"
            f"Total Parameters: {total_params / 1e6:.2f}M\n"
            f"Trainable Parameters: {trainable_params / 1e6:.2f}M"
        )


class TITAN(SlideFeatureExtractor):
    def __init__(self, model_dir: Path, input_size: int = 512):
        self.model_dir = model_dir
        super(TITAN, self).__init__(input_size)
        self.features_dim = 768

    def build_encoders(self):

        cfg = TitanConfig()
        self.slide_encoder = Titan(cfg)

        checkpoint_path = self.model_dir / "titan-slide-encoder.pth"
        print(f"Loading slide encoder weights from {checkpoint_path} ...")
        self.slide_encoder_weights = torch.load(checkpoint_path)
        updated_sd, msg = update_state_dict(
            model_dict=self.slide_encoder.state_dict(),
            state_dict=self.slide_encoder_weights
        )
        self.slide_encoder.load_state_dict(updated_sd, strict=True)
        print(msg)

        print(f"Building tile encoder ...")
        self.tile_encoder, self.eval_transform = self.slide_encoder.return_conch(
            self.model_dir
        )

    def get_transforms(self):
        return self.eval_transform

    def forward_slide(self, tile_features, tile_coordinates, tile_size_lv0):
        tile_features = tile_features.unsqueeze(0)
        tile_coordinates = tile_coordinates.unsqueeze(0)
        output = self.slide_encoder.encode_slide_from_patch_features(
            tile_features, tile_coordinates, tile_size_lv0
        )
        return output



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




class PRISM(SlideFeatureExtractor):
    """
    Slide-level feature extractor (PRISM model).
    """
    def __init__(self, model_dir: Path, input_size = 224):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(PRISM, self).__init__(input_size)


    def build_encoders(self):
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
        self.slide_encoder = Prism(cfg)

        checkpoint_path = self.model_dir / "prism-slide-encoder.pth"
        print(f"Loading slide encoder weights from {checkpoint_path}...")
        self.slide_encoder_weights = torch.load(checkpoint_path, map_location=self.device)
        updated_sd, msg = update_state_dict(
            model_dict=self.slide_encoder.state_dict(), state_dict=self.slide_encoder_weights
        )
        print(msg)
        self.slide_encoder.load_state_dict(updated_sd, strict=True)
        self.slide_encoder.to(self.device)
        self.slide_encoder.eval()

        print(f"Building tile encoder ...")
        self.tile_encoder = Virchow(model_dir=self.model_dir)

    def forward_slide(self,tile_features):
        """Generate slide-level captions from tile embeddings."""
        tile_features = tile_features.unsqueeze(0)
        reprs = self.slide_encoder.slide_representations(tile_features)
        output = reprs["image_embedding"]  # [1, 1280]
        return output

