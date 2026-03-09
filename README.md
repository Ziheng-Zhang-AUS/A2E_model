# A2E: ASR + Translation Pipeline

This repository implements a two-stage pipeline:

1.  **ASR fine-tuning** using Whisper for Wardaman transcription\
2.  **Translation fine-tuning** using Qwen with LoRA via LLaMAFactory

------------------------------------------------------------------------

# Project Structure

A2E_model/ ├── asr/ │ ├── train_whisper.py │ └── requirements.txt │ ├──
translation/ │ ├── configs/ │ │ └── qwen_sft.yaml │ └── requirements.txt
│ ├── data/ │ ├── transcribe/ │ │ ├── train/ │ │ ├── validation/ │ │ └──
test/ │ │ │ └── translate/ │ ├── demo.json │ └── dataset_info.json │ ├──
results/ │ └── whisper_medium/ │ └── README.md

------------------------------------------------------------------------

# ASR (Whisper Fine-tuning)

## Environment Setup

conda create -n whisper_release python=3.10 conda activate
whisper_release pip install -r asr/requirements.txt

CUDA 11.8 or compatible GPU is required.

## Data Format

data/transcribe/ must follow:

data/transcribe/ ├── train/ ├── validation/ └── test/

Each JSON line should contain:

{ "audio": "filename.wav", "text": "transcription" }

## Training

python asr/train_whisper.py

## Output

Fine-tuned checkpoints are saved to:

results/whisper_medium/

------------------------------------------------------------------------

# Translation (Qwen + LoRA via LLaMAFactory)

## Environment Setup

conda create -n llama_release python=3.10 conda activate llama_release
pip install -r translation/requirements.txt

Requires CUDA 12.x compatible GPU.

## Dataset Format

Data must be placed in:

data/translate/

dataset_info.json example:

{ "demo": { "file_name": "demo.json", "formatting": "sharegpt",
"columns": { "messages": "messages" }, "tags": { "role_tag": "role",
"content_tag": "content", "user_tag": "user", "assistant_tag":
"assistant", "system_tag": "system" } } }

Training sample format:

{ "messages": \[ {"role": "system", "content": "..."}, {"role": "user",
"content": "..."}, {"role": "assistant", "content": "..."} \] }

## Training

llamafactory-cli train translation/configs/qwen_sft.yaml

## Output

LoRA checkpoints are saved to:

translation/saves/qwen3-8b-lora/

------------------------------------------------------------------------

# Hardware Requirements

-   NVIDIA GPU
-   ASR: CUDA 11.8+
-   Translation: CUDA 12.x recommended
-   24GB+ GPU memory recommended for Qwen3-8B LoRA training
