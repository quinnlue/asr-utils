import argparse
import io
import json
import os
import tempfile
import time

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr
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
    g.add_argument("--first-n", type=int, default=None,
                    help="Only use the first N samples from the dataset")

    # ── model ──
    g = p.add_argument_group("Model")
    g.add_argument("--model", default="openai/whisper-medium.en",
                    help="Pretrained Whisper model ID")
    g.add_argument("--adapter", default="quinnlue/whisper-med-mlp",
                    help="Optional LoRA adapter path or HF ID")
    g.add_argument("--adapter-subfolder", default="final",
                    help="Subfolder within the adapter repo")

    # ── generation ──
    g = p.add_argument_group("Generation")
    g.add_argument("--batch-size", type=int, default=48)
    g.add_argument("--max-new-tokens", type=int, default=128,
                    help="Maximum new tokens to generate per sample")
    g.add_argument("--max-seq-len", type=int, default=128,
                    help="Max sequence length for the static KV cache")
    g.add_argument("--num-tpus", type=int, default=4,
                    help="Number of TPU cores to use")

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
# Worker (one per TPU core)
# ─────────────────────────────────────────────────────────────

def _worker(index, args, tmp_dir):
    """Runs on each TPU core.  Processes a shard of the dataset and
    writes per-rank results to a temp JSONL file for later merging."""

    device = xm.xla_device()
    rank = index
    world_size = xr.world_size()
    model_dtype = torch.bfloat16 if args.bf16 else torch.float32
    is_main = rank == 0

    # ── Load dataset (no audio decoding) ──
    if is_main:
        print(f"Loading dataset {args.dataset!r} split={args.split!r} …")
    ds = load_dataset(args.dataset, split=args.split)
    ds = ds.cast_column("audio", Audio(decode=False))
    if args.first_n is not None:
        ds = ds.select(range(min(args.first_n, len(ds))))

    # ── Shard across TPU cores ──
    shard_indices = list(range(rank, len(ds), world_size))
    shard = ds.select(shard_indices)
    if is_main:
        print(f"  {len(ds)} total samples, {len(shard)} per core "
              f"({world_size} cores)")

    has_ground_truth = "orthographic_text" in ds.column_names

    # ── Load model (already pre-merged if adapter was specified) ──
    if is_main:
        print(f"Loading model {args.model!r} …")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
    )

    model = model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.processor_id)

    # ── DataLoader (over this core's shard only) ──
    dataloader = DataLoader(
        shard,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ── Inference loop ──
    local_results = []           # list of dicts kept in original index order
    total_batches = (len(shard) + args.batch_size - 1) // args.batch_size
    t_start = time.time()

    if is_main:
        print(f"Running inference ({total_batches} batches/core, "
              f"batch_size={args.batch_size}) …")

    for batch_idx, batch_data in enumerate(dataloader):
        waveforms = batch_data["waveforms"]      # np.ndarray (B, T)
        texts = batch_data["texts"]               # list[str]
        utt_ids = batch_data["utterance_ids"]     # list[str]

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

        # Map back to global indices so we can reassemble in order
        batch_start = batch_idx * args.batch_size
        for j, (uid, pred, ref) in enumerate(
            zip(utt_ids, pred_strs, texts)
        ):
            global_idx = shard_indices[batch_start + j]
            row = {
                "global_idx": global_idx,
                "utterance_id": uid,
                "prediction": pred,
            }
            if has_ground_truth:
                row["reference"] = ref
            local_results.append(row)

        if is_main:
            elapsed = time.time() - t_start
            print(f"  batch {batch_idx + 1}/{total_batches}  "
                  f"({elapsed:.1f}s elapsed)")

    # ── Write per-rank results to temp file ──
    rank_path = os.path.join(tmp_dir, f"rank_{rank}.jsonl")
    with open(rank_path, "w", encoding="utf-8") as f:
        for row in local_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ── Sync all cores before rank 0 merges ──
    xm.rendezvous("inference_done")

    if not is_main:
        return

    # ─────────────────────────────────────────────────
    # Rank 0: merge results from all cores
    # ─────────────────────────────────────────────────
    merged = []
    for r in range(world_size):
        rp = os.path.join(tmp_dir, f"rank_{r}.jsonl")
        with open(rp, "r", encoding="utf-8") as f:
            for line in f:
                merged.append(json.loads(line))

    # Sort back into the original dataset order
    merged.sort(key=lambda x: x["global_idx"])
    # Drop the helper key
    for row in merged:
        row.pop("global_idx", None)

    all_preds = [r["prediction"] for r in merged]
    all_refs = [r.get("reference", "") for r in merged]
    all_ids = [r["utterance_id"] for r in merged]

    # ── stdout ──
    if not args.no_stdout:
        for uid, pred in zip(all_ids, all_preds):
            print(f"  [{uid}] {pred}")

    # ── WER ──
    if has_ground_truth:
        wer = score_wer(actual=all_refs, predicted=all_preds)
        print(f"\nWER: {wer:.4f}")
    else:
        wer = None
        print("\nNo ground-truth text — WER not computed.")

    # ── CSV output ──
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        df = pd.DataFrame(merged)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved CSV → {args.output_csv}")

    # ── JSONL output ──
    if args.output_jsonl:
        os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
        with open(args.output_jsonl, "w", encoding="utf-8") as f:
            for row in merged:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved JSONL → {args.output_jsonl}")

    # ── Summary ──
    elapsed = time.time() - t_start
    print(f"\nDone — {len(all_preds)} samples in {elapsed:.1f}s "
          f"({len(all_preds) / elapsed:.1f} samples/s)")
    if wer is not None:
        print(f"WER: {wer:.4f}")


# ─────────────────────────────────────────────────────────────
# Main (spawns workers)
# ─────────────────────────────────────────────────────────────

def main(args):
    tmp_dir = tempfile.mkdtemp(prefix="whisper_infer_")

    # ── Pre-merge LoRA adapter once on CPU before spawning workers ──
    # Avoids redundant merge_and_unload() across every TPU-core worker.
    if args.adapter is not None:
        model_dtype = torch.bfloat16 if args.bf16 else torch.float32
        print(f"Pre-merging LoRA adapter {args.adapter!r} into "
              f"{args.model!r} on CPU …")
        base = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.model, torch_dtype=model_dtype,
        )
        subfolder_kwargs = (
            {"subfolder": args.adapter_subfolder}
            if args.adapter_subfolder else {}
        )
        base = PeftModel.from_pretrained(base, args.adapter, **subfolder_kwargs)
        base = base.merge_and_unload()

        merged_dir = os.path.join(tmp_dir, "merged_model")
        base.save_pretrained(merged_dir)
        del base
        print("Pre-merge complete — workers will load the merged model.")

        # Keep original model ID for the processor/tokenizer
        args.processor_id = args.model
        # Point workers at the merged checkpoint so they skip adapter loading
        args.model = merged_dir
        args.adapter = None
    else:
        args.processor_id = args.model

    xmp.spawn(_worker, args=(args, tmp_dir))


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main(parse_args())