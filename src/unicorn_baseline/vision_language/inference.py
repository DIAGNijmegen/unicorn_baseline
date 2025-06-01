import sys
from contextlib import nullcontext
from typing import Optional

import torch
import tqdm

from unicorn_baseline.vision.pathology.dataset import TileDataset, TileDatasetFromDisk


def extract_tile_features(model, dataloader, device, mixed_precision):
    """Extracts tile-level features using the given model."""
    features = []
    autocast_context = nullcontext()
    if mixed_precision:
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16)
    with torch.inference_mode():
        with autocast_context:
            for _, image in tqdm.tqdm(
                dataloader,
                desc="Extracting tile-level features",
                unit=" tile",
                leave=True,
                file=sys.stdout,
            ):
                image = image.to(device, non_blocking=True)
                feature = model(image)
                features.append(feature.cpu())
    features = torch.cat(features, dim=0)
    return features.unsqueeze(0)


def get_slide_level_output(model, tile_features, device, mixed_precision):
    tile_features = tile_features.to(device)

    autocast_context = nullcontext()
    if mixed_precision:
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16)

    with torch.inference_mode():
        with autocast_context:
            caption = model.forward_slide(
                tile_features,
            )
    return caption


def generate_caption(
    wsi_path,
    tile_encoder,
    slide_encoder,
    coordinates_dir,
    backend="asap",
    batch_size=1,
    num_workers=4,
    mixed_precision=False,
    load_tiles_from_disk=False,
    tile_format: Optional[str] = None,
):
    tile_encoder.eval().to(tile_encoder.device)
    slide_encoder.eval().to(slide_encoder.device)

    transforms = tile_encoder.get_transforms()

    if load_tiles_from_disk:
        dataset = TileDatasetFromDisk(
            wsi_path,
            tile_dir=coordinates_dir,
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
        tile_encoder, dataloader, tile_encoder.device, mixed_precision
    )

    caption = get_slide_level_output(
        slide_encoder, tile_features, slide_encoder.device, mixed_precision
    )

    return caption
