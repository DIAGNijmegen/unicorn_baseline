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
import orjson
from glob import glob
from pathlib import Path
from typing import Any
import numpy as np


def load_task_description(
    input_path: Path = Path("/input/unicorn-task-description.json"),
) -> dict[str, str]:
    """
    Read information from unicorn-task-description.json
    """
    with open(input_path, "r") as f:
        task_description = json.load(f)
    return task_description


def load_inputs(input_path: Path = Path("/input/inputs.json")) -> list[dict[str, Any]]:
    """
    Read information from inputs.json
    """
    input_information_path = Path(input_path)
    with input_information_path.open("r") as f:
        input_information = json.load(f)

    for item in input_information:
        relative_path = item["interface"]["relative_path"]
        item["input_location"] = Path(f"/input/{relative_path}")

    return input_information


def sanitize_json_content(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json_content(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, np.ndarray)):
        return [sanitize_json_content(v) for v in obj]
    elif isinstance(obj, (str, int, bool, float)):
        return obj
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    else:
        return obj.__repr__()


def write_json_file(*, location, content):
    # Writes a json file with the sanitized content
    content = sanitize_json_content(content)
    with open(location, "wb") as f:
        f.write(orjson.dumps(content))


def resolve_image_path(*, location: str | Path) -> Path:
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )
    if len(input_files) != 1:
        raise ValueError(f"Expected one image file, got {len(input_files)}")

    input_file = Path(input_files[0])
    return input_file


if __name__ == "__main__":
    print(load_inputs("baseline_template/test/input/inputs.json"))
    print(
        load_task_description(
            "baseline_template/test/input/unicorn-task-description.json"
        )
    )
