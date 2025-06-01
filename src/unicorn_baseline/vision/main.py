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
from typing import Any

from unicorn_baseline.vision.pathology.main import run_pathology_vision_task
from unicorn_baseline.vision.radiology.main import run_radiology_vision_task


def run_vision(
    task_description: dict[str, str],
    input_information: list[dict[str, Any]],
    model_dir: Path,
) -> int:
    """
    Process input data
    """
    # retrieve task details
    domain = task_description["domain"]
    task_type = task_description["task_type"]
    task_name = task_description['task_name']

    if domain == "pathology":
        run_pathology_vision_task(
            task_name=task_name,
            task_type=task_type,
            input_information=input_information,
            model_dir=model_dir,
        )
    elif (domain == "CT") | (domain == "MR"):
        run_radiology_vision_task(
            task_type=task_type,
            input_information=input_information,
            model_dir=model_dir,
            domain=domain,
        )
    else:
        raise ValueError(f"Domain '{domain}' not supported yet for vision tasks.")

    return 0
