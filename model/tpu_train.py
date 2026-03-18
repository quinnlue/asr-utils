import os
from model.args import parse_args
from huggingface_hub import login
import wandb
from utils.download_sets import download_model, download_set
os.environ["PJRT_DEVICE"] = "TPU"


import torch_xla.distributed.xla_multiprocessing as xmp

def train_fn(rank, args):
    import io
    import os

    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    import torch_xla.core.xla_model as xm

    from datasets import Audio, load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    from modulations.augment import Augment, AugmentConfig

    device = xm.xla_device()
    print(f"Rank {rank} using device: {device}")

    # ── datasets ──
    print("Loading datasets...")
    ds = load_dataset(args.dataset)
    ds = ds.cast_column("audio", Audio(decode=False))
    noise_ds = load_dataset(args.noise_dataset)
    noise_ds = noise_ds.cast_column("audio", Audio(decode=False))

    train_ds = ds["train"]
    train_ds = train_ds.shuffle(seed=42)

    print(f"Train: {len(train_ds)}")

    # ── processor & model ──
    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    )

    # ── LoRA ──
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

    # Unfreeze layer norms — critical for domain adaptation, adds minimal params
    for name, param in model.named_parameters():
        if "layer_norm" in name or "layernorm" in name:
            param.requires_grad = True

    model.print_trainable_parameters()

    # ── Augmentation pipeline ──
    pipeline = Augment(
        config=AugmentConfig(
            sr=16_000,
            time_stretch_p=args.time_stretch_p,
            time_stretch_min_rate=args.time_stretch_min_rate,
            time_stretch_max_rate=args.time_stretch_max_rate,
            noise_p=args.noise_p,
            noise_snr_db_min=args.noise_snr_db_min,
            noise_snr_db_max=args.noise_snr_db_max,
            noise_peak_limit=args.noise_peak_limit,
            spec_augment_p=args.spec_augment_p,
            spec_policy=args.spec_policy,
            vtlp_p=args.vtlp_p,
            vtlp_alpha_min=args.vtlp_alpha_min,
            vtlp_alpha_max=args.vtlp_alpha_max,
        ),
        noise_ds=noise_ds,
    )

    # ── collate ──
    def collate_fn(batch):
        waveforms = []
        texts = []

        for sample in batch:
            try:
                waveform, sr = sf.read(
                    io.BytesIO(sample["audio"]["bytes"]), dtype="bfloat16"
                )
            except Exception as e:
                print(f"[collate_fn] Skipping broken audio "
                      f"(path={sample['audio'].get('path','?')}): {e}")
                continue
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            if len(waveform) == 0:
                print(f"[collate_fn] Skipping empty audio "
                      f"(path={sample['audio'].get('path','?')})")
                continue
            if sr != 16000:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)

            waveforms.append(waveform)
            texts.append(sample["orthographic_text"])

        model_dtype = torch.bfloat16

        _, input_features = pipeline(waveforms, 16000)
        input_features = torch.from_numpy(input_features).to(model_dtype)

        label_lists = [
            processor.tokenizer(t, truncation=True, max_length=128).input_ids
            for t in texts
        ]
        max_len = max(len(ids) for ids in label_lists)
        padded_labels = [ids + [-100] * (max_len - len(ids)) for ids in label_lists]
        labels = torch.tensor(padded_labels, dtype=torch.long)

        bos_id = processor.tokenizer.bos_token_id
        if bos_id is not None and (labels[:, 0] == bos_id).all():
            labels = labels[:, 1:]

        return {
            "input_features": input_features,
            "labels": labels,
        }

    # ── training arguments ──
    print("Defining training arguments...")
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
        torch_compile=False,  # incompatible with PEFT on XLA; XLA already compiles
        optim=args.optim,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        dataloader_drop_last=True,
    )

    # ── trainer ──
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collate_fn,
        processing_class=processor.feature_extractor,
    )

    # ── resume handling ──
    resume_path = None
    if args.resume_from:
        resume_path = args.resume_from
    else:
        # Auto-detect latest checkpoint in output_dir
        if os.path.isdir(args.output_dir):
            ckpts = [
                d for d in os.listdir(args.output_dir)
                if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, d))
            ]
            if ckpts:
                ckpts.sort(key=lambda d: int(d.split("-")[-1]))
                resume_path = os.path.join(args.output_dir, ckpts[-1])

    if resume_path:
        print(f"▶ Resuming from {resume_path}")
    else:
        print("▶ Starting training from scratch")

    trainer.train(resume_from_checkpoint=resume_path)

    # ── save ──
    if rank == 0:
        final_dir = os.path.join(args.output_dir, "final")
        trainer.save_model(final_dir)
        processor.save_pretrained(final_dir)
        print(f"✅ Final model saved to {final_dir}")

    print(f"✅ Done rank {rank}")

if __name__ == "__main__":
    login()
    wandb.login()
    # Persist the API key in the environment so forked child processes inherit it
    api_key = wandb.api.api_key
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
    args = parse_args()
    download_model(args.model)
    download_set(args.dataset)

    xmp.spawn(train_fn, args=(args,), start_method='fork')
