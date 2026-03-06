import argparse
import io
import os
import sys
import time

import librosa
import numpy as np
import soundfile as sf
import torch

import evaluate
import wandb
from datasets import Audio, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

from augment import Augment, AugmentConfig
from callbacks import (
    AdapterSnapshotCallback,
    PeriodicWERCallback,
    TokenErrorRateTrainer,
)
from score import score_wer


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Fine-tune Whisper with LoRA + augmentations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── data ──
    g = p.add_argument_group("Data")
    g.add_argument("--dataset", default="quinnlue/audio-cleaned",
                    help="HF dataset ID for train/val/test splits")
    g.add_argument("--noise-dataset", default="quinnlue/realclass",
                    help="HF dataset ID for noise samples")
    g.add_argument("--val-size", type=int, default=2400,
                    help="Number of validation samples to use (0 = all)")

    # ── model ──
    g = p.add_argument_group("Model")
    g.add_argument("--model", default="openai/whisper-medium.en",
                    help="Pretrained Whisper model ID")

    # ── LoRA ──
    g = p.add_argument_group("LoRA")
    g.add_argument("--lora-r", type=int, default=64,
                    help="LoRA rank")
    g.add_argument("--lora-alpha", type=int, default=128,
                    help="LoRA alpha")
    g.add_argument("--lora-dropout", type=float, default=0.0,
                    help="LoRA dropout")
    g.add_argument("--lora-target-modules", nargs="+",
                    default=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
                    help="LoRA target modules")

    # ── training ──
    g = p.add_argument_group("Training")
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--grad-accum", type=int, default=2,
                    help="Gradient accumulation steps")
    g.add_argument("--epochs", type=int, default=5)
    g.add_argument("--lr", type=float, default=5e-4,
                    help="Peak learning rate")
    g.add_argument("--lr-scheduler", default="cosine",
                    help="LR scheduler type")
    g.add_argument("--warmup-steps", type=int, default=500)
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--adam-beta1", type=float, default=0.9)
    g.add_argument("--adam-beta2", type=float, default=0.98)
    g.add_argument("--optim", default="adamw_torch",
                    help="Optimizer name")
    g.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True,
                    help="Use bf16 mixed precision")
    g.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                    default=False)
    g.add_argument("--torch-compile", action=argparse.BooleanOptionalAction,
                    default=True, help="Enable torch.compile via Trainer")

    # ── eval / save ──
    g = p.add_argument_group("Evaluation & checkpointing")
    g.add_argument("--eval-steps", type=int, default=500)
    g.add_argument("--save-steps", type=int, default=500)
    g.add_argument("--save-total-limit", type=int, default=32,
                    help="Max checkpoints to keep on disk")
    g.add_argument("--logging-steps", type=int, default=25)
    g.add_argument("--generation-max-length", type=int, default=128)

    # ── I/O & Hub ──
    g = p.add_argument_group("Output & Hub")
    g.add_argument("--output-dir", default="./whisper-medium-finetune",
                    help="Local checkpoint directory")
    g.add_argument("--hub-model-id", default="quinnlue/whisper-medium-finetune",
                    help="HF Hub repo for pushing checkpoints")
    g.add_argument("--push-to-hub", action=argparse.BooleanOptionalAction,
                    default=True)
    g.add_argument("--hub-strategy", default="checkpoint",
                    choices=["checkpoint", "end", "every_save", "all_checkpoints"],
                    help="When to push to the Hub")
    g.add_argument("--report-to", default="wandb",
                    help="Experiment tracker (wandb, tensorboard, none)")

    # ── resume ──
    g = p.add_argument_group("Resumption")
    g.add_argument("--resume-from", default=None,
                    help="Checkpoint name to resume from (e.g. 'last-checkpoint'). "
                         "Omit to train from scratch.")

    # ── callbacks ──
    g = p.add_argument_group("Callbacks")
    g.add_argument("--wer-every-n-steps", type=int, default=100,
                    help="Periodic WER callback frequency")
    g.add_argument("--wer-num-samples", type=int, default=64,
                    help="Samples used for periodic WER check")
    g.add_argument("--adapter-dir", default="./adapters",
                    help="Local dir for adapter snapshots")

    # ── dataloader ──
    g = p.add_argument_group("DataLoader")
    g.add_argument("--dataloader-num-workers", type=int, default=4)
    g.add_argument("--dataloader-pin-memory", action=argparse.BooleanOptionalAction,
                    default=True)
    g.add_argument("--dataloader-persistent-workers", action=argparse.BooleanOptionalAction,
                    default=True)

    # ── augmentation overrides ──
    g = p.add_argument_group("Augmentation")
    g.add_argument("--pitch-shift-p", type=float, default=0.25)
    g.add_argument("--time-stretch-p", type=float, default=0.25)
    g.add_argument("--noise-p", type=float, default=0.5)
    g.add_argument("--spec-augment-p", type=float, default=0.8)
    g.add_argument("--vtlp-p", type=float, default=0.25)

    return p.parse_args(argv)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(args):
    # ── datasets ──
    print("Loading datasets...")
    ds = load_dataset(args.dataset)
    ds = ds.cast_column("audio", Audio(decode=False))
    noise_ds = load_dataset(args.noise_dataset)
    noise_ds = noise_ds.cast_column("audio", Audio(decode=False))

    train_ds = ds["train"]
    val_ds   = ds["validation"]
    if args.val_size > 0:
        val_ds = val_ds.select(range(args.val_size))
    test_ds  = ds["test"]

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    # ── processor & model ──
    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    print("SDPA Attention: ✓ enabled (uses flash kernel automatically on H200)")

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
        if "layer_norm" in name or "layernorm" in name or "conv" in name:
            param.requires_grad = True

    model.print_trainable_parameters()

    # ── Performance optimizations ──
    torch.backends.cudnn.benchmark = True

    # ── Augmentation pipeline ──
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

    # ── collate ──
    def collate_fn(batch, augment=True):
        waveforms = []
        texts = []

        for sample in batch:
            try:
                waveform, sr = sf.read(
                    io.BytesIO(sample["audio"]["bytes"]), dtype="float32"
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

        if len(waveforms) == 0:
            print("[collate_fn] WARNING: entire batch was broken — returning dummy batch")
            dummy_wav = np.zeros(16000, dtype=np.float32)
            waveforms = [dummy_wav]
            texts = [""]

        model_dtype = torch.bfloat16 if args.bf16 else torch.float32

        if augment:
            _, input_features = pipeline(waveforms, 16000)
            input_features = torch.from_numpy(input_features).to(model_dtype)
        else:
            input_features = pipeline.compute_log_mel_batch(waveforms, 16000)
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

    train_collate_fn = lambda batch: collate_fn(batch, augment=True)
    eval_collate_fn  = lambda batch: collate_fn(batch, augment=False)

    # ── shuffle ──
    print("Shuffling train split...")
    train_ds = train_ds.shuffle(seed=42)

    print(f"collate_fn defined (with augmentations).  "
          f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    # ── metrics ──
    print("Defining compute_metrics function...")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str  = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = score_wer(actual=label_str, predicted=pred_str)
        return {"wer": wer}

    print("WER metric ready.")

    # ── training arguments ──
    print("Defining training arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=args.report_to,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        dataloader_persistent_workers=args.dataloader_persistent_workers,
        torch_compile=args.torch_compile,
        optim=args.optim,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_strategy=args.hub_strategy,
    )

    # ── callbacks ──
    wer_callback = PeriodicWERCallback(
        eval_dataset=val_ds,
        collate_fn=eval_collate_fn,
        processor=processor,
        every_n_steps=args.wer_every_n_steps,
        num_samples=args.wer_num_samples,
    )

    adapter_snapshot_callback = AdapterSnapshotCallback(
        hub_repo_id=args.hub_model_id,
        local_dir=args.adapter_dir,
    )

    # ── trainer ──
    trainer = TokenErrorRateTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=train_collate_fn,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
        callbacks=[wer_callback, adapter_snapshot_callback],
    )
    trainer.eval_data_collator = eval_collate_fn

    # ── resume handling ──
    resume_path = None
    if args.resume_from:
        from huggingface_hub import snapshot_download

        local_ckpt = snapshot_download(
            repo_id=args.hub_model_id,
            allow_patterns=f"{args.resume_from}/*",
            local_dir=args.output_dir,
        )
        resume_path = os.path.join(args.output_dir, args.resume_from)
        print(f"▶ Downloaded & resuming from {resume_path}")
    else:
        print("▶ Starting training from scratch")

    trainer.train(resume_from_checkpoint=resume_path)

    # ── post-training evaluation ──
    test_results = trainer.evaluate(eval_dataset=test_ds)
    print(f"Test WER: {test_results['eval_wer']:.4f}")

    # ── save ──
    final_dir = args.output_dir + "-final"
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)

    wandb.finish()
    print("Done! Model saved and W&B run finished.")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main(parse_args())
