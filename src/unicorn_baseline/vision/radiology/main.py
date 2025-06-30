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

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from unicorn_baseline.io import resolve_image_path, write_json_file
from unicorn_baseline.vision.radiology.models.ctfm import encode, load_model
from unicorn_baseline.vision.radiology.models.smalldinov2 import SmallDINOv2
from unicorn_baseline.vision.radiology.patch_extraction import extract_patches
from picai_prep.preprocessing import Sample, PreprocessingSettings
from unicorn_baseline.vision.radiology.models.mrsegmentator import (
    encode_mr,
    load_model_mr,
)


def extract_features_classification(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    input_size: int,
    slug: str,
    max_feature_length: int = 4096,
) -> dict:
    model.eval()
    with torch.no_grad():
        batch = dataset[0]
        scan_volume = batch["image"].to(device)

        patch_features = []
        for d in range(scan_volume.shape[0]):
            slice_ = scan_volume[d].unsqueeze(0).unsqueeze(0)
            slice_resized = F.interpolate(
                slice_,
                size=(input_size, input_size),
                mode="bilinear",
                align_corners=False,
            )
            slice_3ch = slice_resized.repeat(1, 3, 1, 1).to(device)
            feat = model(slice_3ch)
            patch_features.append(feat.squeeze(0).cpu())

        image_level_feature = torch.stack(patch_features).mean(dim=0)
        feature_list = image_level_feature.tolist()[:max_feature_length]

        return {"title": slug, "features": feature_list}


def extract_features_segmentation(
    image,
    model_dir: str,
    domain: str,
    title: str = "patch-level-neural-representation",
    patch_size: list[int] = [16, 64, 64],
    patch_spacing: list[float] | None = None,
) -> list[dict]:
    """
    Generate a list of patch features from a radiology image
    """
    patch_features = []

    image_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        image.GetDirection()
    )
    if (image_orientation != "SPL") and (domain == "CT"):
        image = sitk.DICOMOrient(image, desiredCoordinateOrientation="SPL")

    if (image_orientation != "LPS") and (domain == "MR"):
        image = sitk.DICOMOrient(image, desiredCoordinateOrientation="LPS")

    print(f"Extracting patches from image")
    patches, coordinates, image = extract_patches(
        image=image,
        patch_size=patch_size,
        spacing=patch_spacing,
    )
    if patch_spacing is None:
        patch_spacing = image.GetSpacing()

    if domain == "CT":
        model = load_model(Path(model_dir, "ctfm"))
    if domain == "MR":
        model = load_model_mr(Path(model_dir, "mrsegmentator"))
    print(f"Extracting features from patches")
    for patch, coords in tqdm(
        zip(patches, coordinates), total=len(patches), desc="Extracting features"
    ):
        patch_array = sitk.GetArrayFromImage(patch)
        if domain == "CT":
            features = encode(model, patch_array)
        if domain == "MR":
            features = encode_mr(model, patch_array)
        patch_features.append(
            {
                "coordinates": coords[0],
                "features": features,
            }
        )

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
    domain: str,
):
    # Identify image inputs
    image_inputs = []
    for input_socket in input_information:
        if input_socket["interface"]["kind"] == "Image":
            image_inputs.append(input_socket)

    if task_type == "classification":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = SmallDINOv2(
            model_dir=Path(model_dir, "SmallDINOv2"), use_safetensors=True
        ).to(device)

        outputs = []
        for image_input in image_inputs:
            slug = image_input["interface"]["slug"]
            image_dir = Path(image_input["input_location"])
            scan_path = next(image_dir.glob("*.mha"), None)
            if scan_path is None:
                continue

            from unicorn_baseline.vision.radiology.dataset import get_scan_dataset

            dataset = get_scan_dataset(scan_path, seed=42)

            result = extract_features_classification(
                model=model, dataset=dataset, device=device, input_size=518, slug=slug
            )
            outputs.append(result)

        output_dir = Path("/output")
        output_path = output_dir / "image-neural-representation.json"
        write_json_file(location=output_path, content=outputs)

    elif task_type in ["detection", "segmentation"]:
        output_dir = Path("/output")
        neural_representations = []

        if image_inputs[0]["interface"]["slug"].endswith("prostate-mri"):
            images_to_preprocess = {}
            for image_input in image_inputs:
                image_path = resolve_image_path(location=image_input["input_location"])
                print(f"Reading image from {image_path}")
                image = sitk.ReadImage(str(image_path))

                if "t2" in str(image_input["input_location"]):
                    images_to_preprocess.update({"t2": image})
                if "hbv" in str(image_input["input_location"]):
                    images_to_preprocess.update({"hbv": image})
                if "adc" in str(image_input["input_location"]):
                    images_to_preprocess.update({"adc": image})

            pat_case = Sample(
                scans=[
                    images_to_preprocess.get("t2"),
                    images_to_preprocess.get("hbv"),
                    images_to_preprocess.get("adc"),
                ],
                settings=PreprocessingSettings(
                    spacing=[3, 1.5, 1.5], matrix_size=[16, 256, 256]
                ),
            )
            pat_case.preprocess()

            for image in pat_case.scans:
                neural_representation = extract_features_segmentation(
                    image=image,
                    model_dir=model_dir,
                    domain=domain,
                    title=image_input["interface"]["slug"],
                )
                neural_representations.append(neural_representation)

        else:
            for image_input in image_inputs:
                image_path = resolve_image_path(location=image_input["input_location"])
                print(f"Reading image from {image_path}")
                image = sitk.ReadImage(str(image_path))

                neural_representation = extract_features_segmentation(
                    image=image,
                    model_dir=model_dir,
                    domain=domain,
                    title=image_input["interface"]["slug"],
                )
                neural_representations.append(neural_representation)

        output_path = output_dir / "patch-neural-representation.json"
        write_json_file(location=output_path, content=neural_representations)
