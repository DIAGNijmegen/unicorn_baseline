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
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

import json
import time
from pathlib import Path
from typing import List

import pandas as pd
from dragon_baseline.main import DragonBaseline
from llm_extractinator import extractinate

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def task16_preprocessing(text_parts):
    return "Roman numeral: " + text_parts[0] + "\n\nText:" + text_parts[1]


def setup_folder_structure(basepath: Path, test_data: pd.DataFrame):
    basepath.mkdir(exist_ok=True)
    (basepath / "data").mkdir(exist_ok=True)
    (basepath / "output").mkdir(exist_ok=True)
    (basepath / "tasks").mkdir(exist_ok=True)

    test_data.to_json(basepath / "data" / "test.json", orient="records")


def wait_for_predictions(
    runpath: Path, task_name: str, timeout: int = 300, interval: int = 10
):
    """
    Wait for the predictions to be generated and saved.

    Args:
        timeout (int): Maximum time to wait in seconds.
        interval (int): Interval between checks in seconds.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        for folder in runpath.iterdir():
            if task_name in folder.name:
                print(f"Predictions found in {folder}. Proceeding to postprocess.")
                return folder
        print("Waiting for predictions to complete...")
        time.sleep(interval)
    raise TimeoutError(f"Predictions not found within {timeout} seconds.")


def drop_keys_except(data: List, keys: List[str]) -> List:
    """
    Drop all keys from the dictionary except the specified keys.
    """
    return [
        {key: value for key, value in example.items() if key in keys}
        for example in data
    ]


def post_process_predictions(data: json, task_config):
    task_id = task_config.jobid
    prediction_name = task_config.target.prediction_name
    if prediction_name in (
        "single_label_binary_classification",
        "single_label_regression",
    ):
        for example in data:
            example[prediction_name] = float(example[prediction_name])
    if task_id == 15:
        for example in data:
            example[prediction_name] = [
                example.pop("left"),
                example.pop("right"),
            ]
    elif task_id == 16:
        for example in data:
            keys = [
                "biopsy",
                "cancer",
                "high_grade_dysplasia",
                "hyperplastic_polyps",
                "low_grade_dysplasia",
                "non_informative",
                "serrated_polyps",
            ]
            example[prediction_name] = [
                1.0 if example.pop(key) in ["True", True] else 0.0 for key in keys
            ]
    elif task_id == 17:
        for example in data:
            example[prediction_name] = [
                float(example.pop("lesion_1")),
                float(example.pop("lesion_2")),
                float(example.pop("lesion_3")),
                float(example.pop("lesion_4")),
                float(example.pop("lesion_5")),
            ]
    data = drop_keys_except(data, ["uid", prediction_name])
    return data


def run_language():
    # Read the task configuration, few-shots and test data
    # We'll leverage the DRAGON baseline algorithm for this
    algorithm = DragonBaseline()
    algorithm.load()
    algorithm.analyze()  # needed for verification of predictions
    task_config = algorithm.task
    few_shots = algorithm.df_train
    test_data = algorithm.df_test
    basepath = Path("/opt/app/unicorn_baseline/src/unicorn_baseline/language/")
    setup_folder_structure(basepath, test_data)
    print(f"Task description: {task_config}")

    task_name = task_config.task_name
    task_id = task_config.jobid

    # Task specific preprocessing
    if "Task16_" in task_name:
        test_data["text"] = test_data["text_parts"].apply(task16_preprocessing)

    # Perform data extraction
    extractinate(
        task_id=task_id,
        model_name="phi4",
        num_examples=0,
        temperature=0.0,
        max_context_len="max",
        num_predict=512,
        translate=False,
        data_dir=basepath / "data",
        output_dir=basepath / "output",
        task_dir=basepath / "tasks",
        n_runs=1,
        verbose=False,
        run_name="run",
        reasoning_model=False,
        seed=42,
    )

    # Wait for the predictions to be generated and saved
    runpath = basepath / "output" / "run"
    datafolder = wait_for_predictions(
        runpath=runpath,
        task_name=str(task_id),
        timeout=300,
        interval=10,
    )

    # Load the predictions
    datapath = datafolder / "nlp-predictions-dataset.json"
    with open(datapath, "r") as file:
        predictions = json.load(file)

    predictions = post_process_predictions(predictions, task_config)

    # Save the predictions
    test_predictions_path = OUTPUT_PATH / "nlp-predictions-dataset.json"
    write_json_file(
        location=test_predictions_path,
        content=predictions,
    )

    # Verify the predictions
    algorithm.test_predictions_path = test_predictions_path
    algorithm.verify_predictions()

    print(f"Saved neural representation to {test_predictions_path}")
    return 0


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))
