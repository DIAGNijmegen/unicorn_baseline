#  Copyright 2025 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import torch
import torch.nn as nn
import timm
import safetensors.torch
import os


class SmallDINOv2(nn.Module):
    def __init__(self, model_dir, use_safetensors=True):
        """
        Load DINOv2 ViT-S/14 model from local files using timm.

        Args:
            model_dir (str): Path to the directory containing the model weights.
            use_safetensors (bool): Whether to load from .safetensors (default) or .bin.
        """
        super(SmallDINOv2, self).__init__()

        # Load DINOv2 model architecture from timm
        self.model = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=False,  # Do not download weights online
            num_classes=0,  # Remove classifier
        )

        # Determine which weight file to use
        weights_path = os.path.join(
            model_dir, "model.safetensors" if use_safetensors else "pytorch_model.bin"
        )
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found in {weights_path}")

        # Load weights
        if use_safetensors:
            state_dict = safetensors.torch.load_file(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location="cpu")

        # Load weights into model (allow missing keys since classifier is removed)
        self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model weights from {weights_path}")

        # Freeze all parameters (acts as a fixed feature extractor)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Extracts patch-level features using DINOv2.

        Args:
            x (torch.Tensor): Input tensor of shape [B, 3, H, W] (single patch at a time).

        Returns:
            torch.Tensor: Extracted CLS token features [B, 384].
        """
        features = self.model.forward_features(x)  # Extract raw features

        # CLS token is usually at index 0 in ViT outputs
        return features[:, 0, :]  # Extract the CLS token representation
