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
mkdir -p model/biogpt
echo "Downloading BioGPT model from Hugging Face..."

python3 - <<EOF
from transformers import BioGptTokenizer, BioGptForCausalLM

model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")

model.save_pretrained("model/biogpt")
tokenizer.save_pretrained("model/biogpt")
EOF
echo "✅ Done."
echo ""

# CT-FM
mkdir -p model/ctfm
echo "Downloading CT-FM model..."

python3 - <<EOF
from lighter_zoo import SegResNet

model = SegResNet.from_pretrained(
    "project-lighter/whole_body_segmentation",
    force_download=True
)
model.save_pretrained("model/ctfm")
EOF
echo "✅ Done."
echo ""

# MRSegmentator
mkdir -p model/mrsegmentator
echo "Downloading MRSegmentator model from GitHub..."
curl -L -o weights.zip https://github.com/hhaentze/MRSegmentator/releases/download/v1.2.0/weights.zip && unzip -o weights.zip -d model/mrsegmentator/
echo "✅ Done."
echo ""

# opus-mt-en-nl
mkdir -p model/opus-mt-en-nl
echo "Downloading opus-mt-en-nl model from Hugging Face..."

python3 - <<EOF
from transformers import MarianMTModel, MarianTokenizer
MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-nl").save_pretrained("model/opus-mt-en-nl")
MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-nl").save_pretrained("model/opus-mt-en-nl")
EOF
echo "✅ Done."
echo ""

# phi4
mkdir -p model/phi4
echo "Downloading phi4 model from Ollama..."

export OLLAMA_MODELS="model/phi4"

# --- Start Ollama server if not running ---
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama server..."
    nohup ollama serve > /dev/null 2>&1 &
fi

# --- Wait until Ollama is ready (up to 60s) ---
echo "Waiting for Ollama to become available..."
for i in {1..60}; do
    if ollama list > /dev/null 2>&1; then
        echo "Ollama is ready."
        break
    fi
    sleep 1
done

if ! ollama list > /dev/null 2>&1; then
    echo "❌ Ollama server did not start in time."
    exit 1
fi

# Pull model
ollama pull phi4
echo "✅ Done."
echo ""

# PRISM
echo "Downloading PRISM model from Hugging Face..."
curl -L -H "$HF_HEADER" https://huggingface.co/paige-ai/Prism/resolve/main/model.safetensors -o model/prism-slide-encoder.pth
echo "✅ Done."
echo ""

# TITAN
echo "Downloading TITAN model from Hugging Face..."

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

echo "✅ Done."
echo ""

# Virchow
echo "Downloading Virchow model from Hugging Face..."
curl -L -H "$HF_HEADER" https://huggingface.co/paige-ai/Virchow/blob/main/config.json $HF_AUTH -o model/virchow-config.json
curl -L -H "$HF_HEADER" https://huggingface.co/paige-ai/Virchow/blob/main/pytorch_model.bin $HF_AUTH -o model/virchow-tile-encoder.pth

echo ""
echo "✅ All model weights downloaded and organized."