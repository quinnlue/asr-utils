import argparse
import io
import json
import os
import time

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch_xla.core.xla_model as xm
from tqdm import tqdm
from datasets import Audio, load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from model.tpu_generate import generate
from utils.score import score_wer


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run Whisper inference on TPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── data ──
    g = p.add_argument_group("Data")
    g.add_argument("--dataset", default="quinnlue/audio-cleaned-test",
                    help="HF dataset ID")
    g.add_argument("--split", default="test",
                    help="Dataset split to run inference on")

    # ── model ──
    g = p.add_argument_group("Model")
    g.add_argument("--model", default="openai/whisper-medium.en",
                    help="Pretrained Whisper model ID")
    g.add_argument("--adapter", default=None,
                    help="Optional LoRA adapter path or HF ID")

    # ── generation ──
    g = p.add_argument_group("Generation")
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--max-new-tokens", type=int, default=128,
                    help="Maximum new tokens to generate per sample")
    g.add_argument("--max-seq-len", type=int, default=128,
                    help="Max sequence length for the static KV cache")

    # ── precision ──
    g = p.add_argument_group("Precision")
    g.add_argument("--bf16", action=argparse.BooleanOptionalAction,
                    default=True, help="Use bfloat16")

    # ── output ──
    g = p.add_argument_group("Output")
    g.add_argument("--output-csv", default=None,
                    help="Path to write CSV results")
    g.add_argument("--output-jsonl", default=None,
                    help="Path to write JSONL results")
    g.add_argument("--no-stdout", action="store_true", default=False,
                    help="Suppress printing transcriptions to stdout")

    return p.parse_args(argv)


# ─────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────

def collate_fn(batch):
    """Decode raw audio bytes, resample to 16 kHz, pad, and stack.

    Follows the same byte-level decoding scheme as train.py
    (``Audio(decode=False)`` + ``sf.read(io.BytesIO(...))``) but does
    **not** compute mel spectrograms — that is handled inside
    ``tpu_generate.generate()`` via the processor.

    Returns a dict with:
        waveforms    – np.ndarray (B, T) padded float32 waveforms
        texts        – list[str]  ground-truth transcriptions (may be empty strings)
        utterance_ids – list[str]  per-sample IDs (empty string when absent)
    """
    waveforms = []
    texts = []
    utterance_ids = []

    for sample in batch:
        # ── decode from raw bytes ──
        try:
            waveform, sr = sf.read(
                io.BytesIO(sample["audio"]["bytes"]), dtype="float32"
            )
        except Exception as e:
            print(f"[collate_fn] Skipping broken audio "
                  f"(path={sample['audio'].get('path', '?')}): {e}")
            continue

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if len(waveform) == 0:
            print(f"[collate_fn] Skipping empty audio "
                  f"(path={sample['audio'].get('path', '?')})")
            continue
        if sr != 16_000:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16_000)

        waveforms.append(waveform)
        texts.append(sample.get("orthographic_text", ""))
        utterance_ids.append(sample.get("utterance_id", ""))

    # ── fallback for fully-broken batches ──
    if len(waveforms) == 0:
        print("[collate_fn] WARNING: entire batch was broken — returning dummy batch")
        waveforms = [np.zeros(16_000, dtype=np.float32)]
        texts = [""]
        utterance_ids = [""]

    # ── pad to longest and stack into (B, T) ──
    max_len = max(len(w) for w in waveforms)
    padded = np.zeros((len(waveforms), max_len), dtype=np.float32)
    for i, w in enumerate(waveforms):
        padded[i, : len(w)] = w

    return {
        "waveforms": padded,
        "texts": texts,
        "utterance_ids": utterance_ids,
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(args):
    device = xm.xla_device()
    model_dtype = torch.bfloat16 if args.bf16 else torch.float32

    # ── Load dataset (no audio decoding) ──
    print(f"Loading dataset {args.dataset!r} split={args.split!r} …")
    ds = load_dataset(args.dataset, split=args.split)
    ds = ds.cast_column("audio", Audio(decode=False))
    print(f"  {len(ds)} samples")

    has_ground_truth = "orthographic_text" in ds.column_names

    # ── Load model ──
    print(f"Loading model {args.model!r} …")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
    )

    if args.adapter is not None:
        print(f"Loading LoRA adapter {args.adapter!r} …")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()

    model = model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model)

    # ── DataLoader ──
    dataloader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ── Inference loop ──
    all_preds = []
    all_refs = []
    all_ids = []

    total_batches = (len(ds) + args.batch_size - 1) // args.batch_size
    t_start = time.time()

    print(f"Running inference ({total_batches} batches, "
          f"batch_size={args.batch_size}) …")

    for batch_idx, batch_data in tqdm(enumerate(dataloader)):
        waveforms = batch_data["waveforms"]      # np.ndarray (B, T)
        texts = batch_data["texts"]               # list[str]
        utt_ids = batch_data["utterance_ids"]     # list[str]

        # ── generate ──
        with torch.no_grad():
            token_ids = generate(
                model=model,
                processor=processor,
                batch=waveforms,
                max_new_tokens=args.max_new_tokens,
                device=device,
                max_seq_len=args.max_seq_len,
            )

        pred_strs = processor.tokenizer.batch_decode(
            token_ids.cpu(), skip_special_tokens=True
        )

        all_preds.extend(pred_strs)
        all_refs.extend(texts)
        all_ids.extend(utt_ids)

        # ── per-batch stdout ──
        if not args.no_stdout:
            for uid, pred in zip(utt_ids, pred_strs):
                print(f"  [{uid}] {pred}")

        elapsed = time.time() - t_start
        print(f"  batch {batch_idx + 1}/{total_batches}  "
              f"({elapsed:.1f}s elapsed)")

    # ── WER ──
    if has_ground_truth:
        wer = score_wer(actual=all_refs, predicted=all_preds)
        print(f"\nWER: {wer:.4f}")
    else:
        wer = None
        print("\nNo ground-truth text — WER not computed.")

    # ── Build results table ──
    records = []
    for uid, pred, ref in zip(all_ids, all_preds, all_refs):
        row = {"utterance_id": uid, "prediction": pred}
        if has_ground_truth:
            row["reference"] = ref
        records.append(row)

    # ── CSV output ──
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        df = pd.DataFrame(records)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved CSV → {args.output_csv}")

    # ── JSONL output ──
    if args.output_jsonl:
        os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
        with open(args.output_jsonl, "w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved JSONL → {args.output_jsonl}")

    # ── Summary ──
    elapsed = time.time() - t_start
    print(f"\nDone — {len(all_preds)} samples in {elapsed:.1f}s "
          f"({len(all_preds) / elapsed:.1f} samples/s)")
    if wer is not None:
        print(f"WER: {wer:.4f}")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main(parse_args())