# UNICORN Basline 🦄

Welcome to the official baseline repository for the [UNICORN challenge](https://unicorn.grand-challenge.org/)!<br>
This repository provides reference implementations and tools for tackling a wide range of vision, language, and vision-language tasks in computational pathology and radiology.

[![PyPI version](https://img.shields.io/pypi/v/unicorn-baseline)](https://pypi.org/project/unicorn-baseline/)

This baseline uses the following publicly available foundation models:

- [TITAN](https://huggingface.co/MahmoodLab/TITAN) (link to [publication](https://arxiv.org/abs/2411.19666))
- [Virchow](https://huggingface.co/paige-ai/Virchow)  (link to [publication](https://www.nature.com/articles/s41591-024-03141-0))
- [PRISM](https://huggingface.co/paige-ai/Prism) (link to [publication](https://arxiv.org/abs/2405.10254))
- [MRSegmentator](https://github.com/hhaentze/MRSegmentator/tree/master) (link to [publication](https://arxiv.org/pdf/2405.06463))
- [CT-FM: Whole Body Segmetation](https://github.com/project-lighter/CT-FM) (link to [publication](https://arxiv.org/pdf/2501.09001))

## 🚀 Quickstart

System requirements: Linux-based OS (e.g., Ubuntu 22.04) with Python 3.10+ and Docker installed.<br>
We provide scripts to automate the local testing process using public few-shot data from Zenodo.

### 1. Clone the Repository

```bash
git clone https://github.com/DIAGNijmegen/unicorn_baseline.git
cd unicorn_baseline
```
### 2. Download Model Weights

Download the required model weights and place them in the `model/` directory.

```bash
./download_weights.sh
```

### 3. Build the Docker Container

```bash
./do_build.sh
```

### 4. Perform test run(s)

Make sure to always take the **latest** version of the data on Zenodo.

- **Single Task:** Downloads and prepares data for a single task, then runs the docker on one case.
   ```bash
   ./run_task.sh "https://zenodo.org/records/15315589/files/Task01_classifying_he_prostate_biopsies_into_isup_scores.zip"
   ```
- **All Tasks:** Runs the docker on all supported UNICORN tasks, sequentially.
   ```bash
    ./run_all_tasks.sh
   ```
- **Targeted Test Run:** Run the docker on a specific case folder.
   ```bash
  ./do_test_run.sh path/to/case/folder [docker_image_tag]
  ```

### 5. Save the Container for Submission

```bash
./do_save.sh
```

## 📝 Input & Output Interfaces

- **Input:**
  Each task provides a `unicorn-task-description.json` describing the required inputs and metadata. See [example-data/](example-data/README.md) for sample files and structure.
- **Output:**
  The baseline generates standardized output files (e.g., `image-neural-representation.json`, `patch-neural-representation.json`) as required by the challenge.

## 🧑‍💻 Development

- Source code is organized by modality:
  - [`src/unicorn_baseline/vision/`](src/unicorn_baseline/vision/)
  - [`src/unicorn_baseline/language/`](src/unicorn_baseline/language/)
  - [`src/unicorn_baseline/vision_language/`](src/unicorn_baseline/vision_language/)
- The main entrypoint for inference is [`src/unicorn_baseline/inference.py`](src/unicorn_baseline/inference.py).


## 📜 License

This project is licensed under the [Apache License 2.0](LICENSE).