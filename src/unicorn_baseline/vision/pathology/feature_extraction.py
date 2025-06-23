import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import tqdm

from unicorn_baseline.vision.pathology.dataset import TileDataset, TileDatasetFromDisk


def extract_tile_features(
    *,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    batch_size: int,
    device: torch.device,
    use_mixed_precision: bool,
):
    """Extracts tile-level features using the given model."""
    features = []
    autocast_context = nullcontext()
    if use_mixed_precision:
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16)
    with torch.inference_mode():
        with autocast_context:
            for _, image in tqdm.tqdm(
                dataloader,
                desc="Extracting tile-level features",
                unit=" tile",
                unit_scale=batch_size,
                leave=True,
                file=sys.stdout,
            ):
                image = image.to(device, non_blocking=True)
                feature = model(image)
                features.append(feature.cpu())
    return torch.cat(features, dim=0)


def aggregate_slide_features(
    *,
    model: nn.Module,
    tile_features: torch.Tensor,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    use_mixed_precision: bool,
):
    """Aggregates tile-level features into a slide-level embedding."""
    tile_features = tile_features.to(device)
    coordinates = torch.tensor(dataset.coordinates, dtype=torch.int64, device=device)
    autocast_context = nullcontext()
    if use_mixed_precision:
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16)
    with torch.inference_mode():
        with autocast_context:
            wsi_feature = model.forward_slide(tile_features)
    return wsi_feature.squeeze(0).cpu().tolist()


def extract_features(
    *,
    wsi_path: Path,
    model: nn.Module,
    coordinates_dir: Path,
    task_type: str,
    backend: str = "asap",
    batch_size: int = 1,
    num_workers: int = 4,
    use_mixed_precision: bool = False,
    load_tiles_from_disk: bool = False,
    tile_dir: Optional[str] = None,
    tile_format: Optional[str] = None,
):
    """Main function to extract features for all cases."""

    model.eval().to(model.device)
    transforms = model.get_transforms()

    if load_tiles_from_disk:
        dataset = TileDatasetFromDisk(
            wsi_path,
            tile_dir=tile_dir,
            tile_format=tile_format,
            transforms=transforms,
        )
    else:
        dataset = TileDataset(
            wsi_path,
            coordinates_dir=coordinates_dir,
            backend=backend,
            transforms=transforms,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )

    tile_features = extract_tile_features(
        model=model,
        dataloader=dataloader,
        batch_size=batch_size,
        device=model.device,
        use_mixed_precision=use_mixed_precision,
    )

    if task_type in ["classification", "regression"]:
        # aggregate to a single slide-level feature
        slide_feature = aggregate_slide_features(
            model=model,
            tile_features=tile_features,
            dataset=dataset,
            device=model.device,
            use_mixed_precision=use_mixed_precision,
        )
        return slide_feature
    else:
        # return tile-level features for detection/segmentation tasks
        return tile_features
