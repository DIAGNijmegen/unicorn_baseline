#!/bin/bash

# ------------------------------------------------------------------------------
# Bash script to simplify local development using public few-shot data 
# for a single task from Zenodo.
#
# Steps performed by this script:
#   1. Checks if the task data is already unzipped locally.
#   2. If not, checks if the zip file has already been downloaded.
#   3. If not, downloads the zip file from Zenodo and unzips it to the local data directory.
#   4. Performs a local test run (`./do_test_run.sh`) for an example shot.
#
# Example for Task 1:
#   BASE_URL="/path/to/latest/version/of/zenodo/public-shots"
#   ZIP="Task01_classifying_he_prostate_biopsies_into_isup_scores.zip"
#
# Usage:
#   ./run_task.sh "${BASE_URL}/${ZIP}"
# ------------------------------------------------------------------------------

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <ZIP_FILE_URL>"
    exit 1
fi

FILE_URL="$1"
OUTPUT_NAME=$(basename "${FILE_URL%%\?*}")  
TASK_NAME="${OUTPUT_NAME%.zip}"
TEMP_ZIP="${OUTPUT_NAME}.part"
LOCAL_DATA_DIR="local_data"
TASK_FOLDER="${LOCAL_DATA_DIR}/${TASK_NAME}"

# Ensure the local_data directory exists
if [ ! -d "$LOCAL_DATA_DIR" ]; then
    mkdir "$LOCAL_DATA_DIR"
    echo "Created directory: $LOCAL_DATA_DIR"
fi

# 1. Check if already unzipped
if [ -d "${TASK_FOLDER}/shots-public" ]; then
    echo "Data already extracted at ${TASK_FOLDER}/shots-public, nothing to do."
else
    # 2. If ZIP file doesn't exist or is invalid, (re)download it to a temp file
    if [ ! -f "$OUTPUT_NAME" ]; then
        echo "Downloading $OUTPUT_NAME from $FILE_URL to $TEMP_ZIP ..."
        curl -L -o "$TEMP_ZIP" "$FILE_URL"

        # Validate ZIP
        if unzip -tq "$TEMP_ZIP" >/dev/null; then
            mv "$TEMP_ZIP" "$OUTPUT_NAME"
            echo "ZIP file validated and renamed to $OUTPUT_NAME"
        else
            echo "ERROR: Downloaded ZIP file is corrupt. Removing $TEMP_ZIP"
            rm -f "$TEMP_ZIP"
            exit 1
        fi
    fi

    # 3. Extract
    echo "Extracting $OUTPUT_NAME to $TASK_FOLDER ..."
    mkdir -p "$TASK_FOLDER"
    unzip -oq "$OUTPUT_NAME" -d "$TASK_FOLDER"
    echo "Unzipped to $TASK_FOLDER"

    # 4. Remove ZIP after successful extraction
    rm -f "$OUTPUT_NAME"
    echo "Removed ZIP file: $OUTPUT_NAME"
fi

## Depending on the modality the test test_run is receives a single public shot (vision/-language) or all shots (language) as input
TASK_NAME=$(basename "$TASK_FOLDER")

# Extract just the task number (e.g., "12" from "Task12_predicting_histopathology_sample_origin.zip")
TASK_NUMBER=$(echo "$TASK_NAME" | grep -oP 'Task\K[0-9]+')

# List of language tasks 
LANGUAGE_TASKS=(12 13 14 15 16 17 18 19)

# Check if task number is in the LANGUAGE_TASKS list
IS_LANGUAGE_TASK=false
for LANG_TASK in "${LANGUAGE_TASKS[@]}"; do
    if [ "$TASK_NUMBER" -eq "$LANG_TASK" ]; then
        IS_LANGUAGE_TASK=true
        break
    fi
done

# Mount data folder
if [ "$IS_LANGUAGE_TASK" = true ]; then
    echo "Detected language task (Task${TASK_NUMBER}). Mounting the entire task folder."
    ABS_PATH=$(realpath "${TASK_FOLDER}/shots-public")
else
    echo "Detected vision/-language task (Task${TASK_NUMBER}). Mounting the first case from shots-public."
    SHOTS_PUBLIC="${TASK_FOLDER}/shots-public"
    FIRST_CASE=$(ls "$SHOTS_PUBLIC" | head -n 1)
    ABS_PATH=$(realpath "${SHOTS_PUBLIC}/${FIRST_CASE}")

fi

# Run local test
./do_test_run.sh "$ABS_PATH"
