import os
import argparse
import torch
import evaluate

from dataclasses import dataclass
from typing import List, Dict, Any, Union

from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    Seq2SeqTrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="openai/whisper-medium")
    parser.add_argument("--language", type=str, default="su")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--max_steps", type=int, default=300)
    return parser.parse_args()


metric = evaluate.load("wer")


def prepare_dataset(batch, feature_extractor, tokenizer):
    audio = batch["audio"]

    batch["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def compute_metrics(pred, tokenizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    valid_preds, valid_refs = [], []

    for p, r in zip(pred_str, label_str):
        if p.strip() and r.strip():
            valid_preds.append(p)
            valid_refs.append(r)

    if len(valid_preds) == 0:
        return {"wer": None}

    wer = 100 * metric.compute(predictions=valid_preds, references=valid_refs)
    return {"wer": wer}


def main():
    args = parse_args()

    train_json = os.path.join(args.data_dir, "train.jsonl")
    val_json = os.path.join(args.data_dir, "val.jsonl")
    test_json = os.path.join(args.data_dir, "test.jsonl")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name)

    processor = WhisperProcessor.from_pretrained(
        args.model_name,
        language=args.language,
        task="transcribe"
    )

    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model.enable_input_require_grads()

    dataset = load_dataset(
        "json",
        data_files={
            "train": train_json,
            "validation": val_json,
            "test": test_json,
        }
    )

    def build_path_mapper(split_name):
        def add_prefix(batch):
            filename = batch["audio"]
            full_path = os.path.join(args.data_dir, split_name, filename)
            batch["audio"] = full_path
            return batch
        return add_prefix

    for split in dataset.keys():
        dataset[split] = dataset[split].map(build_path_mapper(split))

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    dataset = dataset.map(
        lambda x: prepare_dataset(x, feature_extractor, tokenizer),
        remove_columns=dataset["train"].column_names,
    )

    model.generation_config.language = "sundanese"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-5,
        warmup_steps=30,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        eval_steps=10,
        save_steps=10,
        logging_steps=1,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=3,
        report_to="none",
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=processor,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
    )

    print("Starting training...")
    trainer.train()
    print("Done.")


if __name__ == "__main__":
    main()