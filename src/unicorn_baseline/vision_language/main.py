from pathlib import Path

from unicorn_baseline.io import resolve_image_path, write_json_file
from unicorn_baseline.vision.pathology.main import extract_coordinates, save_coordinates
from unicorn_baseline.vision.pathology.wsi import FilterParams, TilingParams
from unicorn_baseline.vision_language.inference import generate_caption
from unicorn_baseline.vision_language.models import PRISM, Virchow


def get_file_path(file_location, extensions):
    for ext in extensions:
        potential_path = Path(f"{file_location}{ext}")
        if potential_path.exists():
            return potential_path
    return None


def save_output(caption, name):
    output_dict = [{"uid": name, "text": caption}]
    slug = "nlp-predictions-dataset"
    output_path = Path("/output")
    output_filename = output_path / f"{slug}.json"

    write_json_file(
        location=output_filename,
        content=output_dict,
    )

    print(f"Caption saved to {output_filename}")


def run_vision_language_task(*, input_information, model_dir):

    tissue_mask_path = None
    for input_socket in input_information:
        if input_socket["interface"]["kind"] == "Image":
            image_name = input_socket["image"]["name"]
            wsi_path = resolve_image_path(location=input_socket["input_location"])
        elif input_socket["interface"]["kind"] == "Segmentation":
            tissue_mask_path = resolve_image_path(location=input_socket["input_location"])

    num_workers = 4
    batch_size = 32
    mixed_precision = True
    max_number_of_tiles = 14000
    tiling_params = TilingParams(spacing=0.5, tolerance=0.07, tile_size=224, overlap=0.0, drop_holes=False, min_tissue_percentage=0.25, use_padding=True)
    filter_params = FilterParams(ref_tile_size=256, a_t=4, a_h=2, max_n_holes=8)

    # create output directories
    coordinates_dir = Path("/tmp/coordinates/")
    coordinates_dir.mkdir(parents=True, exist_ok=True)
    coordinates, _, level, resize_factor, tile_size_lv0, image_spacing, image_size = (
        extract_coordinates(
            wsi_path=wsi_path,
            tissue_mask_path=tissue_mask_path,
            tiling_params=tiling_params,
            filter_params=filter_params,
            max_number_of_tiles=max_number_of_tiles,
            num_workers=num_workers,
        )
    )

    save_coordinates(
        wsi_path=wsi_path,
        coordinates=coordinates,
        tile_level=level,
        tile_size=tiling_params.tile_size,
        resize_factor=resize_factor,
        tile_size_lv0=tile_size_lv0,
        target_spacing=tiling_params.spacing,
        save_dir=coordinates_dir,
    )

    tile_encoder = Virchow(model_dir=model_dir)
    prism = PRISM(model_dir=model_dir)

    caption = generate_caption(
        wsi_path,
        tile_encoder,
        prism,
        coordinates_dir,
        backend="asap",
        batch_size=batch_size,
        num_workers=num_workers,
        mixed_precision=mixed_precision,
    )

    caption = caption[0].replace("</s>", "").strip()

    save_output(caption, image_name)
