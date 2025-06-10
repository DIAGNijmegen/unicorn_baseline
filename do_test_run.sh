#!/bin/bash

# ============================================================================
# do_test_run.sh
#
# Performs local test run using the specified Docker container on a single test case per task.
#
# USAGE:
#   ./do_test_run.sh <CASE_FOLDER> [DOCKER_IMAGE_TAG]
#
# DESCRIPTION:
#   - This script performs a test run on one public few-shot case using the Docker.
#   - The first argument should point to a fully prepared input folder
#     (typically a subdirectory of shots-public/, e.g., `shots-public/123456`).
#   - The second optional argument specifies the Docker image tag (default: unicorn_baseline).
#
# EXAMPLES:
#   ./do_test_run.sh local_data/Task01_classifying_he_prostate_biopsies_into_isup_scores/shots-public/0f958c8bbbc828b2e043e49ea39e16e2

#
# ARGUMENTS:
#   $1 - Path to input case folder (required)
#   $2 - Docker image tag to run (optional, default: unicorn_baseline)

# ============================================================================
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# === Argument parsing ===
if [ $# -lt 1 ]; then
    echo "Error: Missing required argument <CASE_FOLDER_OR_ZIP>"
    echo "Usage: $0 <CASE_FOLDER_OR_ZIP> [DOCKER_IMAGE_TAG]"
    exit 1
fi

INPUT_DIR="$1"
DOCKER_IMAGE_TAG="${2:-unicorn_baseline}"


echo "Using DOCKER_IMAGE_TAG: $DOCKER_IMAGE_TAG"

OUTPUT_DIR="${SCRIPT_DIR}/test/output"
DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"

echo "=+= (Re)build the container"

source "${SCRIPT_DIR}/do_build.sh" "$DOCKER_IMAGE_TAG"
# Ensure permissions are set correctly on the output
# This allows the host user (e.g. you) to access and handle these files
cleanup() {
    echo "=+= Cleaning permissions ..."

    docker run --rm \
      --platform=linux/amd64 \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG \
      -c "chmod -R -f o+rwX /output/* || true"
}

if [ -d "$OUTPUT_DIR" ]; then
  chmod -f o+rwx "$OUTPUT_DIR"
  echo "=+= Cleaning up any earlier output"
  # Use the container itself to circumvent ownership problems
  docker run --rm \
      --platform=linux/amd64 \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG \
      -c "rm -rf /output/* || true"
else
  mkdir -p -m o+rwx "$OUTPUT_DIR"
fi

trap cleanup EXIT

echo "=+= Doing a forward pass"
## Note the extra arguments that are passed here:
# '--network none'
#    entails there is no internet connection
# 'gpus all'
#    enables access to any GPUs present
# '--volume <NAME>:/tmp'
#   is added because on Grand Challenge this directory cannot be used to store permanent files
# '-volume ../model:/opt/ml/model/:ro'
#   is added to provide access to the (optional) tarball-upload locally
docker volume create "$DOCKER_NOOP_VOLUME" > /dev/null
docker run --rm \
    --platform=linux/amd64 \
    --network none \
    --gpus all \
    --volume "$INPUT_DIR":/input:ro \
    --volume "$OUTPUT_DIR":/output \
    --volume "$DOCKER_NOOP_VOLUME":/tmp \
    --volume "${SCRIPT_DIR}/model":/opt/ml/model/:ro \
    "$DOCKER_IMAGE_TAG"
docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null

echo "=+= Wrote results to ${OUTPUT_DIR}"
echo "=+= Save this image for uploading via ./do_save.sh \"${DOCKER_IMAGE_TAG}\""