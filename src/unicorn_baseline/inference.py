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

"""
The following is the UNICORN baseline.

It is meant to run within a container.

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

import torch
import torchvision

torchvision.disable_beta_transforms_warning()

from pathlib import Path

from unicorn_baseline.io import load_inputs, load_task_description
from unicorn_baseline.language.main import run_language
from unicorn_baseline.vision.main import run_vision
from unicorn_baseline.vision_language.main import run_vision_language_task

INPUT_PATH = Path("/data/temporary/unicorn/debugging/encoding/vision/task-10/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path("/data/temporary/unicorn/models/baseline-models/ctfm")

def print_directory_contents(path: Path | str):
    path = Path(path)
    for child in path.iterdir():
        if child.is_dir():
            print_directory_contents(child)
        else:
            print(child)


def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"- number of devices: {torch.cuda.device_count()}")
        print(f"- current device: { (current_device := torch.cuda.current_device())}")
        print(f"- properties: {torch.cuda.get_device_properties(current_device).name}")
    print("=+=" * 10)


def run_vision_and_visionlanguage(input_dir: Path, model_dir: Path) -> int:
    """
    Process input data
    """

    task_description = load_task_description(
        input_path=input_dir / "unicorn-task-description.json"
    )
    input_information = load_inputs(input_path=input_dir / "inputs.json")

    # retrieve task details
    domain = task_description["domain"]
    modality = task_description["modality"]
    task_type = task_description["task_type"]

    if modality == "vision":
        run_vision(
            task_description=task_description,
            input_information=input_information,
            model_dir=model_dir,
        )
    elif modality == "vision-language":
        run_vision_language_task(
            input_information=input_information,
            model_dir=model_dir,
        )
    else:
        raise ValueError(
            f"Modality '{modality}' and domain '{domain}' not supported yet"
        )

    return 0


def run():
    # show GPU information
    _show_torch_cuda_info()

    # print contents of input folder
    print("input folder contents:")
    print_directory_contents(INPUT_PATH)
    print("=+=" * 10)

    # check if the task is image or text
    if (INPUT_PATH / "nlp-task-configuration.json").exists():
        return run_language(OUTPUT_PATH)
    else:
        return run_vision_and_visionlanguage(INPUT_PATH, MODEL_PATH)


if __name__ == "__main__":

    raise SystemExit(run())
