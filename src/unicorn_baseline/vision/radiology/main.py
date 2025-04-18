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

import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from unicorn_baseline.io import resolve_image_path, write_json_file
# from unicorn_baseline.vision.radiology.ctfm.ctfm import encode, load_model
from unicorn_baseline.vision.radiology.models import SmallDINOv2
from unicorn_baseline.vision.radiology.patch_extraction import extract_patches


# Choose your output path for classification
CLASSIFICATION_OUTPUT_PATH = Path("/output/features.json")


def extract_features_classification(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    input_size: int,
    max_feature_length: int = 4096,
):
    model.eval()
    with torch.no_grad():
        batch = dataset[0]
        case_id = batch["ID"]
        scan_volume = batch["image"].to(device)

        logging.info(f"Processing scan: {case_id}...")
        start_time = time.time()

        patch_features = []
        for d in range(scan_volume.shape[0]):
            slice_ = scan_volume[d].unsqueeze(0).unsqueeze(0)
            slice_resized = F.interpolate(
                slice_, size=(input_size, input_size), mode="bilinear", align_corners=False
            )
            slice_3ch = slice_resized.repeat(1, 3, 1, 1).to(device)
            feat = model(slice_3ch)
            patch_features.append(feat.squeeze(0).cpu())

        image_level_feature = torch.stack(patch_features).mean(dim=0)
        feature_list = image_level_feature.tolist()[:max_feature_length]

        output_json = {
            "title": f"images/{"chest-ct-region-of-interest-cropout"}",
            "features": feature_list
        }

        with open(CLASSIFICATION_OUTPUT_PATH, "w") as f:
            json.dump(output_json, f, indent=4)

        elapsed = time.time() - start_time
        logging.info(f"Saved to {CLASSIFICATION_OUTPUT_PATH} (Time: {elapsed:.2f}s)")

def extract_features_segmentation(
    image_path: Path,
    model_dir: str,
    title: str = "patch-level-neural-representation",
    patch_size: list[int] = [224, 224, 16],
    patch_spacing: list[float] | None = None,
) -> list[dict]:
    """
    Generate a list of patch features from a radiology image
    """
    patch_features = []
    print(f"Reading image from {image_path}")
    image = sitk.ReadImage(str(image_path))

    print(f"Extracting patches from image")
    patches, coordinates = extract_patches(
        image=image,
        patch_size=patch_size,
        spacing=patch_spacing,
    )
    if patch_spacing is None:
        patch_spacing = image.GetSpacing()

    model = load_model(model_dir + '/modelweights/ct_fm.pth')
    print(f"Extracting features from patches")
    for patch, coords in tqdm(zip(patches, coordinates), total=len(patches), desc="Extracting features"):
        patch_array = sitk.GetArrayFromImage(patch)
        features = encode(model, patch_array)
        patch_features.append({
            "coordinates": coords[0],
            "features": features,
        })

    patch_level_neural_representation = make_patch_level_neural_representation(
        patch_features=patch_features,
        patch_size=patch_size,
        patch_spacing=patch_spacing,
        image_size=image.GetSize(),
        image_origin=image.GetOrigin(),
        image_spacing=image.GetSpacing(),
        image_direction=image.GetDirection(),
        title=title,
    )
    return patch_level_neural_representation

def make_patch_level_neural_representation(
    *,
    title: str,
    patch_features: Iterable[dict],
    patch_size: Iterable[int],
    patch_spacing: Iterable[float],
    image_size: Iterable[int],
    image_spacing: Iterable[float],
    image_origin: Iterable[float] = None,
    image_direction: Iterable[float] = None,
) -> dict:
    if image_origin is None:
        image_origin = [0.0] * len(image_size)
    if image_direction is None:
        image_direction = np.identity(len(image_size)).flatten().tolist()
    return {
        "meta": {
            "patch-size": list(patch_size),
            "patch-spacing": list(patch_spacing),
            "image-size": list(image_size),
            "image-origin": list(image_origin),
            "image-spacing": list(image_spacing),
            "image-direction": list(image_direction),
        },
        "patches": list(patch_features),
        "title": title,
    }


def run_radiology_vision_task(
    *,
    task_type: str,
    input_information: dict[str, Any],
    model_dir: Path,
):
    # Identify image inputs
    image_inputs = []
    for input_socket in input_information:
        if input_socket["interface"]["kind"] == "Image":
            image_inputs.append(input_socket)

    if task_type == "classification":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        logging.info(f"Using device: {gpu_name}")

        # Initialize DINO model
        model = SmallDINOv2(model_dir="./resources", use_safetensors=True).to(device)

        # Each image directory should have a .mha
        for image_input in image_inputs:
            image_dir = Path(image_input["input_location"])
            scan_path = next(image_dir.glob("*.mha"), None)
            if scan_path is None:
                logging.warning("No .mha file found in the specified directory.")
                continue

            from unicorn_baseline.vision.radiology.dataset import get_scan_dataset
            dataset = get_scan_dataset(scan_path, seed=42)

            # Run classification feature extraction
            extract_features_classification(
                model,
                dataset,
                device=device,
                input_size=518
            )

    elif task_type in ["detection", "segmentation"]:
        output_dir = Path("/output")
        neural_representations = []
        for image_input in image_inputs:
            image_path = resolve_image_path(location=image_input["input_location"])
            neural_representation = extract_features_segmentation(
                image_path=image_path,
                model_dir=model_dir,
                title=image_input["interface"]["slug"]
            )
            neural_representations.append(neural_representation)

        output_path = output_dir / "patch-neural-representation.json"
        write_json_file(location=output_path, content=neural_representations)


if __name__ == "__main__":
    run_radiology_vision_task()
