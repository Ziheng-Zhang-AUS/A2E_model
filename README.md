# A2E

This repository implements a two-stage pipeline:

1. **ASR fine-tuning** using Whisper for Wardaman transcription
2. **Translation fine-tuning** using Qwen with LoRA via LLaMAFactory



# **1. ASR (Whisper Fine-tuning)**

## **Environment Setup**

```
conda create -n whisper_release python=3.10
conda activate whisper_release
pip install -r asr/requirements.txt
```



## **Data Format**

The directory structure must be:

```
data/transcribe/
├── train.jsonl
├── val.jsonl
├── test.jsonl
├── train/
├── validation/
└── test/
```

Each JSON line file should follow this format:

```
{ "audio": "filename.wav", "text": "transcription" }
```



## Training

```bash
python asr/train_whisper.py \
  --data_dir data/transcribe \
  --model_name openai/whisper-medium \
  --language su \
  --output_dir results/whisper_medium \
  --max_steps 300
```



## **Output**

Fine-tuned Whisper checkpoints are saved to:

```
result/
```





# **2. Translation (Qwen + LoRA via LLaMAFactory)**



## **Environment Setup**

```python
conda create -n llama_release python=3.10
conda activate llama_release
pip install -r translation/requirements.txt
```



## **Dataset Format**

Place translation data in:

```
data/translate/
```

A required dataset_info.json example:

```
{
  "demo": {
    "file_name": "demo.json",
    "formatting": "sharegpt",
    "columns": { "messages": "messages" },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  }
}
```

Each training example should follow:

```
{
 "messages": [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
 ]
}
```



## **Training**

```
llamafactory-cli train translation/configs/qwen_sft.yaml
```

