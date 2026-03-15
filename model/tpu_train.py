import argparse
import io
import os
import sys
import time

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

import wandb
from datasets import Audio, load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Seq2SeqTrainingArguments,
)
from transformers import Trainer

from modulations.augment import Augment, AugmentConfig
from model.args import parse_args


def main(args):
    # -------------- DATASETS --------------
    print("Loading datasets...")
    ds = load_dataset(args.dataset)
    ds = ds.cast_column("audio", Audio(decode=False))
    train_ds = ds["train"].shuffle(seed=42)


    noise_ds = load_dataset(args.noise_dataset)
    noise_ds = noise_ds.cast_column("audio", Audio(decode=False))

    pipeline = Augment(
        config=AugmentConfig(
            sr=16_000,
            pitch_shift_p=args.pitch_shift_p,
            pitch_min_semitones=-4.0,
            pitch_max_semitones=2.0,
            time_stretch_p=args.time_stretch_p,
            time_stretch_min_rate=0.8,
            time_stretch_max_rate=1.25,
            time_stretch_leave_length_unchanged=False,
            noise_p=args.noise_p,
            noise_snr_db_min=5.0,
            noise_snr_db_max=30.0,
            noise_peak_limit=0.99,
            spec_augment_p=args.spec_augment_p,
            spec_policy="LB",
            vtlp_p=args.vtlp_p,
            vtlp_alpha_min=0.8,
            vtlp_alpha_max=1.2,
        ),
        noise_ds=noise_ds,
    )


    # -------------- PROCESSOR & MODEL --------------
    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
    )

    # -------------- LoRA --------------
    print("Loading LoRA config...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if "layer_norm" in name or "layernorm" in name:
            param.requires_grad = True

    model.print_trainable_parameters()


    # -------------- COLLATE FUNCTION --------------


    _start       = model.config.decoder_start_token_id              # <|startoftranscript|>
    _lang_token  = processor.tokenizer.convert_tokens_to_ids("<|en|>")
    _task_token  = processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
    _notimestamp = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    PREFIX = [_start, _lang_token, _task_token, _notimestamp]

    def collate_fn(batch, augment=True):
        waveforms = []
        texts = []

        for sample in batch:
            waveform, sr = sf.read(
                io.BytesIO(sample["audio"]["bytes"]), dtype="float32"
            )
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            if sr != 16000:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
            waveforms.append(waveform)
            texts.append(sample["orthographic_text"])

        max_label_len = 128
        prefix_len = len(PREFIX)
        label_lists = [
            PREFIX + processor.tokenizer(
                t, truncation=True, max_length=max_label_len - prefix_len
            ).input_ids
            for t in texts
        ]
        padded_labels = [ids + [-100] * (max_label_len - len(ids)) for ids in label_lists]
        labels = torch.tensor(padded_labels, dtype=torch.long)

        if augment:
            _, input_features = pipeline(waveforms, 16000)
            input_features = torch.from_numpy(input_features).to(torch.bfloat16)
        else:
            input_features = pipeline.compute_log_mel_batch(waveforms, 16000)
            input_features = torch.from_numpy(input_features).to(torch.bfloat16)

        return {
            "input_features": input_features,
            "labels": labels,
        }


    train_collate_fn = lambda batch: collate_fn(batch, augment=True)


    # -------------- TRAINING ARGUMENTS --------------
    print("Defining training arguments...")
    # torch.compile can rename wrapped parameter paths (e.g. _orig_mod.*) and
    # break Accelerate's PEFT parameter mapping on TPU.
    is_tpu_runtime = os.environ.get("PJRT_DEVICE", "").upper() == "TPU"
    use_torch_compile = args.torch_compile and not is_tpu_runtime
    if args.torch_compile and is_tpu_runtime:
        print("Disabling torch_compile on TPU to avoid PEFT/Accelerate param mapping issues.")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        dataloader_persistent_workers=args.dataloader_persistent_workers,
        torch_compile=use_torch_compile,
        optim=args.optim,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
    )

    # -------------- TRAINER --------------
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=train_collate_fn,
    )

    trainer.train()
    trainer.save_model(args.output_dir + "-final")
    processor.save_pretrained(args.output_dir + "-final")

    wandb.finish()
    print("Done! Model saved and W&B run finished.")

if __name__ == "__main__":
    args = parse_args()
    main(args)