# UNICORN Basline 🦄

This repository contains the baseline code for various tasks in the [UNICORN challenge](https://unicorn.grand-challenge.org/).

[![PyPI version](https://img.shields.io/pypi/v/unicorn-baseline)](https://pypi.org/project/unicorn-baseline/)

## Repository Structure 🗂️

The repository is structured as follows:
```
unicorn_baseline
├── src/unicorn_baseline
│   ├── vision/             # Code for vision tasks
│   ├── language/           # Code for language tasks
│   ├── vision_language/    # Code for vision-language tasks
│   └── inference.py        # Entrypoint script
├── example-data/           # Examples of interfaces and sample files
└── Dockerfile              # Example docker file
```

## 🚀 Getting Started

System requirements: Linux-based OS (e.g., Ubuntu 22.04) with Python 3.10+ and Docker installed.

1. [Local development with Docker using public shots from Zenodo](./setup-docker.md).

## Further information 

The models used in this baseline are: 
Radiology-Vision: 
- [MRSegmentator](https://github.com/hhaentze/MRSegmentator/tree/master): [Paper](https://arxiv.org/pdf/2405.06463) - [Weights](https://github.com/hhaentze/MRSegmentator/releases/tag/v1.2.0)
- [CT-FM: Whole Body Segmetation](https://github.com/project-lighter/CT-FM): [Paper](https://arxiv.org/pdf/2501.09001) - [Weights](https://huggingface.co/project-lighter/whole_body_segmentation)
