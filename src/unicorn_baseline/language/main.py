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
import time
from pathlib import Path
from typing import List

import pandas as pd
from dragon_baseline.main import DragonBaseline
from llm_extractinator import extractinate

from unicorn_baseline.io import write_json_file
from unicorn_baseline.language.task_definitions import TASK_DEFINITIONS


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


def generate_task_definition(config: DragonBaseline):
    """
    Combines shared config and task-specific information.

    Parameters:
        config (DragonBaseline): Object with attributes: jobid, input_name

    Returns:
        dict: The full task definition
    """
    task_id = f"Task{int(config.jobid):03}"  # e.g., 12 -> "Task012"

    if task_id not in TASK_DEFINITIONS:
        raise ValueError(f"Unknown task ID: {task_id}")

    base = TASK_DEFINITIONS[task_id].copy()
    base.update(
        {
            "Input_Field": config.input_name,
            "Data_Path": "test.json",
        }
    )
    return base


def generate_task_file(config: DragonBaseline, task_folder: Path):
    """
    Generates a task file based on the task ID and configuration.

    Parameters:
        task_id (str): e.g., "17"
        config (dict): dict with keys: 'Input_Field', 'Label_Field', 'Data_Path'
        output_path (Path): Path to save the generated task file
    """
    task_definition = generate_task_definition(config)
    output_path = task_folder / f"Task{int(config.jobid):03}.json"
    with open(output_path, "w") as f:
        json.dump(task_definition, f, indent=4)


def post_process_predictions(data: json, task_config: DragonBaseline):
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


def run_language(OUTPUT_PATH: Path) -> int:
    # Read the task configuration, few-shots and test data
    # We'll leverage the DRAGON baseline algorithm for this
    algorithm = DragonBaseline()
    algorithm.load()
    algorithm.analyze()  # needed for verification of predictions
    task_config = algorithm.task
    few_shots = algorithm.df_train
    test_data = algorithm.df_test
    basepath = Path("/opt/app/workdir/language")
    setup_folder_structure(basepath, test_data)
    print(f"Task description: {task_config}")

    task_name = task_config.task_name
    task_id = task_config.jobid

    generate_task_file(
        config=task_config,
        task_folder=basepath / "tasks",
    )

    # Task specific preprocessing
    if task_name.startswith("Task16_"):
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
