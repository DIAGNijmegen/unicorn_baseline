#!/bin/bash

set -e

# Prompt for HuggingFace token if not set
if [ -z "$HF_TOKEN" ]; then
    if [ -t 0 ]; then
        read -s -p "Enter your Hugging Face API token (input will not be visible): " HF_TOKEN
        echo
        export HF_TOKEN
    fi
fi

HF_HEADER=""
if [ -n "$HF_TOKEN" ]; then
    HF_HEADER="Authorization: Bearer $HF_TOKEN"
fi

mkdir -p model

# SmallDINOv2

# biogpt

# ctfm

# mrsegmentator

# opus-mt-en-nl

# phi4

# PRISM
curl -L -H "$HF_HEADER" https://huggingface.co/paige-ai/Prism/resolve/main/model.safetensors -o model/prism-slide-encoder.pth

# TITAN
python3 - <<EOF
import torch
from transformers import AutoModel

slide_encoder = AutoModel.from_pretrained(
    "MahmoodLab/TITAN", trust_remote_code=True
)
slide_encoder_sd = slide_encoder.state_dict()
tile_encoder, _ = slide_encoder.return_conch()
tile_encoder_sd = tile_encoder.state_dict()
torch.save(slide_encoder_sd, "model/titan-slide-encoder.pth")
torch.save(tile_encoder_sd, "model/conch-tile-encoder.pth")
EOF

# Virchow
curl -L -H "$HF_HEADER" https://huggingface.co/paige-ai/Virchow/blob/main/config.json $HF_AUTH -o model/virchow-config.json
curl -L -H "$HF_HEADER" https://huggingface.co/paige-ai/Virchow/blob/main/pytorch_model.bin $HF_AUTH -o model/virchow-tile-encoder.pth


echo "All model weights downloaded and organized."