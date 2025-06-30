from typing import Iterable
import numpy as np
import SimpleITK as sitk
from picai_prep.preprocessing import resample_img, crop_or_pad
import itertools


def pad_image(image: sitk.Image, patch_size: Iterable[int]) -> sitk.Image:
    """
    Pads the input image symmetrically so its dimensions become divisible by the specified patch size.

    Padding is performed using the minimum intensity value found in the original image, ensuring padded regions blend naturally with background values.

    Args:
        image (sitk.Image): Input 3D image to be padded.
        patch_size (list[int]): Desired patch size as [x, y, z]. After padding, each dimension of the image will be divisible by the corresponding element in this list.

    Returns:
        sitk.Image: Padded SimpleITK image with dimensions divisible by `patch_size`.
    """
    min_val = float(sitk.GetArrayFromImage(image).min())
    size = image.GetSize()
    pad = [(p - s % p) % p for s, p in zip(size, patch_size)]
    new_size = tuple(s + pad_dim for s, pad_dim in zip(size, pad))

    image = crop_or_pad(
        image=image,
        size=new_size[::-1],
        pad_only=True,
    )

    return image


def extract_patches(
    image: sitk.Image,
    patch_size: Iterable[int],
    spacing: Iterable[float] | None = None,
) -> tuple[list[sitk.Image], list[tuple]]:
    """
    Extracts uniformly sized patches from a 3D SimpleITK image, optionally resampling it to a specified voxel spacing before extraction.

    If `spacing` is provided, the image is first resampled using linear interpolation to achieve the specified spacing. The image is then padded so that its dimensions become exactly divisible by the given patch size. Patches are extracted systematically, covering the entire image volume without overlap or gaps.

    Args:
        image (sitk.Image): Input 3D image from which to extract patches.
        patch_size (list[int]): Patch size as [x, y, z], defining the dimensions of each extracted patch.
        spacing (list[float] | None, optional): Desired voxel spacing as [x, y, z]. If provided, the image will be resampled to this spacing before patch extraction. Defaults to None.

    Returns:
        - patches (list[sitk.Image]): List of extracted image patches.
        - coordinates (list[tuple]): List of coordinates for each patch, formatted as:
            ((x_start, y_start, z_start), (x_end, y_end, z_end)) in world coordinates.
            Each coordinate pair represents the start and end points of the patch in the original image's physical space.
    """
    if spacing is not None:
        # resample image to specified spacing
        image = resample_img(
            image=image,
            out_spacing=spacing[::-1],
            interpolation=sitk.sitkLinear,
        )
        print(f"Resampled image to spacing: {spacing}. Image size: {image.GetSize()}")

    # pad image to fit patch size
    image = pad_image(
        image=image,
        patch_size=patch_size,
    )

    # generate patch coordinates (x, y, z)
    image_size = image.GetSize()
    steps = [range(0, image_size[dim], patch_size[dim]) for dim in range(3)]

    patches = []
    coordinates = []
    for x, y, z in itertools.product(*steps):
        start_coords = (x, y, z)
        patch = sitk.RegionOfInterest(image, patch_size, start_coords)
        patches.append(patch)
        matrix_coordinates = (
            (x, y, z),
            (x + patch_size[0], y + patch_size[1], z + patch_size[2]),
        )
        world_coordinates = tuple(
            image.TransformIndexToPhysicalPoint(coord) for coord in matrix_coordinates
        )
        coordinates.append(world_coordinates)

    return patches, coordinates, image
