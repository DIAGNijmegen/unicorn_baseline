import random
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk
import torch

# === Processing Settings ===
SIZE_MM = 70
SIZE_PX = 518

# === Training Parameters ===
SEED = 42
NUM_WORKERS = 8

# === Data Augmentation ===
ROTATION = ((-20, 20), (-20, 20), (-20, 20))
TRANSLATION = True


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_augmentation(slice_2d):
    if ROTATION:
        angles = [np.random.uniform(*ROTATION[i]) for i in range(3)]
        slice_2d = ndi.rotate(
            slice_2d, angle=angles[0], axes=(0, 1), reshape=False, mode="nearest"
        )
    if TRANSLATION:
        shift_values = np.random.uniform(-5, 5, size=2)
        slice_2d = ndi.shift(slice_2d, shift=shift_values, mode="nearest")
    return slice_2d


def clip_and_scale(npzarray, maxHU=400.0, minHU=-1000.0):
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray = np.clip(npzarray, 0, 1)
    return npzarray


class CTScanDataset(torch.utils.data.Dataset):
    def __init__(self, scan_path: Path, seed: int = 42):
        self.SIZE_MM = SIZE_MM
        self.SIZE_PX = SIZE_PX
        self.SEED = SEED
        self.NUM_WORKERS = NUM_WORKERS

        set_seed(seed)
        self.scan_path = scan_path
        self.case_id = scan_path.stem

        image_sitk = sitk.ReadImage(str(scan_path))
        self.img = clip_and_scale(sitk.GetArrayFromImage(image_sitk))
        self.img = np.array([apply_augmentation(slice_2d) for slice_2d in self.img])

    def __getitem__(self, idx):
        return {
            "image": torch.from_numpy(self.img).float(),
            "ID": self.case_id,
            "image_path": str(self.scan_path),
        }

    def __len__(self):
        return 1


def get_scan_dataset(scan_path: Path, seed: int = 42):
    return CTScanDataset(scan_path, seed)
