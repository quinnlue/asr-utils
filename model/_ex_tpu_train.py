import torch_xla.distributed.xla_multiprocessing as xmp

def train_fn(rank, flags):
    import os
    os.environ["PJRT_DEVICE"] = "TPU"
    from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
    from datasets import Dataset
    import numpy as np
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    print(f"Rank {rank} using device: {device}")

    # Same dataset/model setup
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    rng = np.random.default_rng(42)
    input_ids = rng.integers(200, tokenizer.vocab_size, (512, 64)).tolist()
    train_ds = Dataset.from_dict({
        "input_ids": input_ids,
        "attention_mask": [[1]*64 for _ in range(512)],
        "labels": input_ids
    })
    train_ds.set_format("torch")

    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=8,
        num_train_epochs=2,
        learning_rate=5e-4,
        optim="adamw_torch",
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        dataloader_drop_last=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
    )

    trainer.train()
    print(f"✅ Done rank {rank}")

# Launch 4 processes for 4 TPU cores
xmp.spawn(train_fn, args=(None,), start_method='fork')