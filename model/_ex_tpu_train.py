#!/usr/bin/env python3
"""train_tpu.py — DistilGPT2 on dummy data across 8 TPU cores."""

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)


def build_dummy_dataset(num_samples=2048, max_length=128, vocab_size=50257):
    """Create a fake token-id dataset."""
    rng = np.random.default_rng(42)
    input_ids = rng.integers(
        low=200, high=vocab_size, size=(num_samples, max_length)
    ).tolist()

    dataset = Dataset.from_dict({
        "input_ids": input_ids,
        "attention_mask": [[1] * max_length for _ in range(num_samples)],
        "labels": input_ids,
    })
    dataset.set_format("torch")
    return dataset


def main():
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    train_ds = build_dummy_dataset(num_samples=2048, max_length=128)
    eval_ds  = build_dummy_dataset(num_samples=256,  max_length=128)

    training_args = TrainingArguments(
        output_dir="./tpu_demo_output",

        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        dataloader_drop_last=True,
        data_loader_pin_memory=False,

        num_train_epochs=3,
        learning_rate=5e-4,
        weight_decay=0.01,
        warmup_steps=30,
        lr_scheduler_type="cosine",

        optim="adamw_torch",

        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,

        bf16=True,
        dataloader_num_workers=0,

        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()
    trainer.save_model("./tpu_demo_output/final")
    print("✅  Training complete!")


if __name__ == "__main__":
    main()