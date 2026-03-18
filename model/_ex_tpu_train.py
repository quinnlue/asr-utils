import os
os.environ["PJRT_DEVICE"] = "TPU"

import wandb
import torch_xla.distributed.xla_multiprocessing as xmp

def train_fn(rank, flags):
    from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
    from datasets import Dataset
    import numpy as np
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    print(f"Rank {rank} using device: {device}")

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

    checkpoint_dir = "/mnt/checkpoints/ex_tpu_train"

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=8,
        num_train_epochs=20,
        learning_rate=5e-4,
        optim="adamw_torch",
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=3,
        bf16=True,
        dataloader_drop_last=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    trainer.train()

    if rank == 0:
        final_dir = os.path.join(checkpoint_dir, "final")
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"✅ Final model saved to {final_dir}")

    print(f"✅ Done rank {rank}")

if __name__ == "__main__":
    # Launch 4 processes for 4 TPU cores
    xmp.spawn(train_fn, args=(None,), start_method='fork')