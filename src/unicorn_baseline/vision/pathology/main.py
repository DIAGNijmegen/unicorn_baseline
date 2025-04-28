import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tqdm
import wholeslidedata as wsd
from PIL import Image

from unicorn_baseline.io import resolve_image_path, write_json_file
from unicorn_baseline.vision.pathology.feature_extraction import extract_features
from unicorn_baseline.vision.pathology.models import TITAN
from unicorn_baseline.vision.pathology.wsi import FilterParams, TilingParams, WholeSlideImage


def extract_coordinates(
    *,
    wsi_path: Path,
    tissue_mask_path: Path,
    spacing,
    tile_size,
    overlap: float = 0.0,
    filter_params: FilterParams = None,
    tiling_params: TilingParams = None,
    max_num_tile: Optional[int] = None,
    num_workers: int = 1,
):
    """
    Extracts tile coordinates from a whole slide image (wsi) based on the given parameters.

    Args:
        wsi_path (str): File path to the wsi.
        tissue_mask_path (str): File path to the tissue mask associated with the wsi.
        spacing (float): Desired spacing for the tiles, in micron per pixel (mpp).
        tile_size (int): Desired size of the tiles to extract.
        overlap (float, optional): Overlap between tiles. Defaults to 0.0.
        max_num_tile (Optional[int], optional): Maximum number of tiles to keep. If None, all tiles are kept. Defaults to None.
        num_workers (int, optional): Number of workers to use for parallel processing. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - sorted_coordinates (list): List of tile coordinates sorted by tissue percentage.
            - sorted_tissue_percentages (list): List of tissue percentages corresponding to the sorted coordinates.
            - tile_level (int): Tile level used for extraction.
            - resize_factor (float): Resize factor applied to the tiles.
            - tile_size_lv0 (int): Tile size at level 0 in the wsi.
    """
    wsi = WholeSlideImage(wsi_path, tissue_mask_path)
    (
        coordinates,
        tissue_percentages,
        tile_level,
        resize_factor,
        tile_size_lv0,
    ) = wsi.get_tile_coordinates(
        spacing,
        tile_size,
        overlap=overlap,
        filter_params=filter_params,
        tiling_params=tiling_params,
        num_workers=num_workers,
    )

    image_spacings = wsi.spacings[0]
    required_level = wsi.get_best_level_for_spacing(spacing)
    image_size = wsi.level_dimensions[required_level]

    # sort coordinates by tissue percentage
    sorted_coordinates, sorted_tissue_percentages = sort_coordinates_with_tissue(
        coordinates=coordinates, tissue_percentages=tissue_percentages
    )
    if max_num_tile is not None:
        sorted_coordinates = sorted_coordinates[:max_num_tile]
        sorted_tissue_percentages = sorted_tissue_percentages[:max_num_tile]
    return (
        sorted_coordinates,
        sorted_tissue_percentages,
        tile_level,
        resize_factor,
        tile_size_lv0,
        image_spacings,
        image_size,
    )


def save_coordinates(
    *,
    wsi_path: Path,
    coordinates,
    tile_level,
    tile_size,
    resize_factor,
    tile_size_lv0,
    target_spacing,
    save_dir: str,
):
    """
    Saves tile coordinates and associated metadata into a .npy file.

    Args:
        wsi_path (Path): File path to the whole slide image (wsi).
        coordinates (list of tuples): List of (x, y) coordinates of tiles, defined with respect to level 0.
        tile_level (int): Level of the image pyramid at which tiles have been extracted.
        tile_size (int): Desired tile size.
        resize_factor (float): Factor by which the tile size must be resized.
        tile_size_lv0 (int): Size of the tile at level 0.
        target_spacing (float): Target spacing for the tiles.
        save_dir (str): Directory where the output .npy file will be saved.

    Returns:
        Path: Path to the saved .npy file containing tile coordinates and associated metadata.

    Notes:
        - The output file is saved with the same name as the wsi file stem.
        - The metadata includes the resized tile size, tile level, resize factor, tile size at level 0,
          and target spacing for each tile.
    """
    wsi_name = wsi_path.stem
    output_path = Path(save_dir, f"{wsi_name}.npy")
    x = [c[0] for c in coordinates]  # defined w.r.t level 0
    y = [c[1] for c in coordinates]  # defined w.r.t level 0
    ntile = len(x)
    tile_size_resized = int(tile_size * resize_factor)

    dtype = [
        ("x", int),
        ("y", int),
        ("tile_size_resized", int),
        ("tile_level", int),
        ("resize_factor", float),
        ("tile_size_lv0", int),
        ("target_spacing", float),
    ]
    data = np.zeros(ntile, dtype=dtype)
    for i in range(ntile):
        data[i] = (
            x[i],
            y[i],
            tile_size_resized,
            tile_level,
            resize_factor,
            tile_size_lv0,
            target_spacing,
        )

    data_arr = np.array(data)
    np.save(output_path, data_arr)
    return output_path


def sort_coordinates_with_tissue(*, coordinates, tissue_percentages):
    """
    Sorts coordinates and their corresponding tissue percentages.

    The function creates mocked filenames by combining the x and y values
    of the coordinates into a string format (e.g., "x_y.jpg"). It then
    sorts the coordinates and tissue percentages based on these mocked
    filenames.

    Args:
        coordinates (list of tuple): A list of tuples representing coordinates,
            where each tuple contains two integers (x, y).
        tissue_percentages (list of float): A list of tissue percentages
            corresponding to the tile for each coordinate.

    Returns:
        tuple: A tuple containing two lists:
            - sorted_coordinates (list of tuple): The coordinates sorted based
              on the mocked filenames.
            - sorted_tissue_percentages (list of float): The tissue
              percentages sorted in the same order as the coordinates.
    """
    # mock region filenames
    mocked_filenames = [f"{x}_{y}.jpg" for x, y in coordinates]
    # combine mocked filenames with coordinates and tissue percentages
    combined = list(zip(mocked_filenames, coordinates, tissue_percentages))
    # sort combined list by mocked filenames
    sorted_combined = sorted(combined, key=lambda x: x[0])
    # extract sorted coordinates and tissue percentages
    sorted_coordinates = [coord for _, coord, _ in sorted_combined]
    sorted_tissue_percentages = [tissue for _, _, tissue in sorted_combined]
    return sorted_coordinates, sorted_tissue_percentages


def save_tile(
    *,
    x: int,
    y: int,
    wsi_path: Path,
    spacing: float,
    tile_size: int,
    resize_factor: int | float,
    save_dir: Path,
    tile_format: str,
    backend: str = "asap",
):
    tile_size_resized = int(tile_size * resize_factor)
    wsi = wsd.WholeSlideImage(wsi_path, backend=backend)
    tile_arr = wsi.get_patch(
        x, y, tile_size_resized, tile_size_resized, spacing=spacing, center=False
    )
    tile = Image.fromarray(tile_arr).convert("RGB")
    if resize_factor != 1:
        tile = tile.resize((tile_size, tile_size))
    tile_fp = save_dir / f"{int(x)}_{int(y)}.{tile_format}"
    tile.save(tile_fp)
    return tile_fp


def save_tile_mp(args):
    coord, wsi_path, spacing, tile_size, resize_factor, tile_dir, tile_format = args
    x, y = coord
    return save_tile(
        x=x, y=y, wsi_path=wsi_path, spacing=spacing, tile_size=tile_size, resize_factor=resize_factor, save_dir=tile_dir, tile_format=tile_format
    )


def save_tiles(
    *,
    wsi_path: Path,
    coordinates: list[tuple[int, int]],
    tile_level: int,
    tile_size: int,
    resize_factor: float,
    save_dir: Path,
    tile_format: str,
    backend: str = "asap",
    num_workers: int = 1,
):
    wsi_name = wsi_path.stem
    wsi = wsd.WholeSlideImage(wsi_path, backend=backend)
    tile_spacing = wsi.spacings[tile_level]
    tile_dir = save_dir / wsi_name
    tile_dir.mkdir(parents=True, exist_ok=True)
    iterable = [
        (coord, wsi_path, tile_spacing, tile_size, resize_factor, tile_dir, tile_format)
        for coord in coordinates
    ]
    with mp.Pool(num_workers) as pool:
        for _ in tqdm.tqdm(
            pool.imap_unordered(save_tile_mp, iterable),
            desc=f"Saving tiles for {wsi_path.stem}",
            unit=" tile",
            total=len(iterable),
            leave=True,
            file=sys.stdout,
        ):
            pass


def save_feature_to_json(
    *,
    feature,
    task_type,
    title,
    coordinates=None,
    tile_size=None,
    spacing=None,
    image_size=None,
    image_spacing=None,
    image_origin=None,
    image_direction=None,
):
    """
    Saves the extracted feature vector to a JSON file in the required format.
    """
    if task_type in ["classification", "regression"]:
        output_dict = [{"title": title, "features": feature}]
        output_path = Path("/output")
        output_filename = output_path / "image-neural-representation.json"

    else:
        print("Spacing: ", spacing)
        if image_origin is None:
            image_origin = [0.0] * len(image_size)
        if image_direction is None:
            image_direction = np.identity(len(image_size)).flatten().tolist()

        output_dict = [
            {
                "title": title,  # Use WSI filename as title
                "patches": [
                    {
                        "coordinates": [int(coord[0]), int(coord[1])],
                        "features": feat.cpu().tolist(),
                    }
                    for coord, feat in zip(coordinates, feature)
                ],
                "meta": {
                    "patch-size": tile_size,
                    "patch-spacing": [spacing, spacing],
                    "image-size": image_size,
                    "image-origin": image_origin,
                    "image-spacing": [image_spacing, image_spacing],
                    "image-direction": image_direction,
                },
            }
        ]

        output_path = Path("/output")
        output_filename = output_path / "patch-neural-representation.json"

    write_json_file(
        location=output_filename,
        content=output_dict,
    )

    print(f"Feature vector saved to {output_filename}")


def run_pathology_vision_task(
    *,
    task_type: str,
    input_information: dict[str, Any],
    model_dir: Path,
):
    tissue_mask_path = None
    for input_socket in input_information:
        if input_socket["interface"]["kind"] == "Image":
            image_title = input_socket["image"]["pk"]
            wsi_path = resolve_image_path(location=input_socket["input_location"])
        elif input_socket["interface"]["kind"] == "Segmentation":
            tissue_mask_path = resolve_image_path(location=input_socket["input_location"])

    target_spacing = 0.5
    use_mixed_precision = True
    save_tiles_to_disk = False
    tile_format = "jpg"
    max_num_tile = 14000

    num_workers = min(mp.cpu_count(), 8)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"]))

    # coonfigurations for tile extraction based on tasks
    clf_config = {
        "batch_size": 32,
        "tile_size": 512,
        "filter_params": FilterParams(ref_tile_size=256, a_t=4, a_h=2, max_n_holes=8),
        "tiling_params": TilingParams(drop_holes=False, tissue_thresh=0.25, use_padding=True),
    }

    detection_segmentation_config = {
        "batch_size": 1,
        "tile_size": 224,
        "filter_params": FilterParams(ref_tile_size=64, a_t=1, a_h=1, max_n_holes=8),
        "tiling_params": TilingParams(drop_holes=False, tissue_thresh=0.1, use_padding=True),
        "overlap": 0.5,
    }

    task_configs = {
        "classification": clf_config,
        "regression": clf_config,
        "detection": detection_segmentation_config,
        "segmentation": detection_segmentation_config,
    }

    config = task_configs[task_type]

    # create output directories
    coordinates_dir = Path("/tmp/coordinates/")
    coordinates_dir.mkdir(parents=True, exist_ok=True)

    # Extract tile coordinates
    coordinates, _, level, resize_factor, tile_size_lv0, image_spacing, image_size = (
        extract_coordinates(
            wsi_path=wsi_path,
            tissue_mask_path=tissue_mask_path,
            spacing=target_spacing,
            tile_size=config["tile_size"],
            overlap=config.get("overlap", 0.0),
            num_workers=num_workers,
            max_num_tile=max_num_tile,
            filter_params=config["filter_params"],
            tiling_params=config["tiling_params"],
        )
    )

    save_coordinates(
        wsi_path=wsi_path,
        coordinates=coordinates,
        tile_level=level,
        tile_size=config["tile_size"],
        resize_factor=resize_factor,
        tile_size_lv0=tile_size_lv0,
        target_spacing=target_spacing,
        save_dir=coordinates_dir,
    )

    tile_dir = None
    if save_tiles_to_disk:
        tile_dir = Path("/tmp/tiles/")
        tile_dir.mkdir(parents=True, exist_ok=True)
        save_tiles(
            wsi_path=wsi_path,
            coordinates=coordinates,
            tile_level=level,
            tile_size=tile_size,
            resize_factor=resize_factor,
            save_dir=tile_dir,
            tile_format=tile_format,
            num_workers=num_workers,
        )

    print("=+=" * 10)
    feature_extractor = TITAN(model_dir)

    # Extract tile or slide features
    feature = extract_features(
        wsi_path=wsi_path,
        model=feature_extractor,
        coordinates_dir=coordinates_dir,
        task_type=task_type,
        backend="asap",
        batch_size=config["batch_size"],
        num_workers=num_workers,
        use_mixed_precision=use_mixed_precision,
        load_tiles_from_disk=save_tiles_to_disk,
        tile_dir=tile_dir,
        tile_format=tile_format,
    )

    if task_type in ["classification", "regression"]:
        save_feature_to_json(feature=feature, task_type=task_type, title=image_title)
    elif task_type in ["detection", "segmentation"]:
        tile_size = [config["tile_size"], config["tile_size"], 3]

        save_feature_to_json(
            feature=feature,
            task_type=task_type,
            title=image_title,
            coordinates=coordinates,
            tile_size=tile_size,
            spacing=target_spacing,
            image_size=image_size,
            image_spacing=image_spacing,
        )
