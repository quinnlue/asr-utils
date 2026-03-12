#!/usr/bin/env python3
"""
TPU inference for Whisper ASR models (with optional LoRA adapters).
Data parallel across chips + batched per chip.

Usage as library:
    from tpu_inference import tpu_transcribe
    df = tpu_transcribe(samples, ...)

Usage as CLI:
    python tpu_inference.py --base-model openai/whisper-medium.en \
        --adapter-repo quinnlue/whisper-medium-ckpts \
        --dataset-repo quinnlue/audio-cleaned --dataset-split validation \
        --num-samples 15700 --batch-size 32 --output-csv records.csv \
        --hf-repo quinnlue/whisper-medium-finetune --hf-path-in-repo val_records.csv
"""

import io
import json
import os
import shutil
import tempfile
import time

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch


# ── Worker (runs on each TPU chip) ───────────────────────────────

def _transcribe_worker(
    index, tmpdir, base_model_name, adapter_repo, adapter_subfolder,
    max_new_tokens, batch_size, dtype_str,
):
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    from peft import PeftModel

    device = torch_xla.device()
    world_size = xr.world_size()
    dtype = getattr(torch, dtype_str)

    # Load metadata to figure out total count & my shard
    with open(os.path.join(tmpdir, "meta.json")) as f:
        meta = json.load(f)
    total = meta["total"]

    my_indices = list(range(index, total, world_size))

    if index == 0:
        print(f"  {world_size} chips | batch={batch_size}/chip | "
              f"{total} samples total")

    # ── Load model ───────────────────────────────────────
    processor = AutoProcessor.from_pretrained(base_model_name)
    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        base_model_name, torch_dtype=dtype,
    )

    if adapter_repo:
        model = PeftModel.from_pretrained(
            base_model, adapter_repo, subfolder=adapter_subfolder or "",
        )
        model = model.merge_and_unload()
    else:
        model = base_model

    model.eval().to(device)
    xm.mark_step()
    xm.wait_device_ops()

    if index == 0:
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        label = f"{base_model_name}"
        if adapter_repo:
            label += f" + {adapter_repo}"
        print(f"  Model loaded ({params_m:.0f}M params, {dtype}): {label}")

    # ── Load my shard of pre-processed audio arrays ──────
    my_arrays = []
    for i in my_indices:
        arr = np.load(os.path.join(tmpdir, f"audio_{i}.npy"))
        my_arrays.append(arr)

    # ── Warmup / XLA compilation ─────────────────────────
    warmup_arrays = my_arrays[:batch_size]
    if len(warmup_arrays) < batch_size:
        warmup_arrays += [warmup_arrays[0]] * (batch_size - len(warmup_arrays))

    features = processor.feature_extractor(
        warmup_arrays, sampling_rate=16000, return_tensors="pt", padding=True,
    )
    input_features = features.input_features.to(device)

    if index == 0:
        print(f"  Compiling (warmup)...")
    t0 = time.time()

    with torch.no_grad():
        _ = model.generate(input_features=input_features, max_new_tokens=max_new_tokens)
    xm.mark_step()
    xm.wait_device_ops()

    if index == 0:
        print(f"  Compiled in {time.time() - t0:.1f}s")

    # ── Process all batches ──────────────────────────────
    if index == 0:
        print(f"  Running inference...")

    xm.wait_device_ops()
    t0 = time.time()

    results = {}
    for batch_start in range(0, len(my_arrays), batch_size):
        batch_arrays = my_arrays[batch_start:batch_start + batch_size]
        batch_indices = my_indices[batch_start:batch_start + batch_size]

        # Pad last batch to keep static shape for XLA
        real_count = len(batch_arrays)
        if real_count < batch_size:
            batch_arrays = batch_arrays + [batch_arrays[0]] * (batch_size - real_count)

        features = processor.feature_extractor(
            batch_arrays, sampling_rate=16000, return_tensors="pt", padding=True,
        )
        input_features = features.input_features.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_features=input_features, max_new_tokens=max_new_tokens,
            )
        xm.mark_step()

        decoded = processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True,
        )

        for j, idx in enumerate(batch_indices):
            results[str(idx)] = decoded[j].strip()

        if index == 0:
            done = min(batch_start + batch_size, len(my_arrays))
            print(f"    chip 0: {done}/{len(my_arrays)}", flush=True)

    xm.wait_device_ops()
    elapsed = time.time() - t0

    with open(os.path.join(tmpdir, f"results_{index}.json"), "w") as f:
        json.dump(results, f)

    if index == 0:
        print(f"  Inference done in {elapsed:.1f}s "
              f"({len(my_arrays)} samples on this chip)")


# ── Public API ───────────────────────────────────────────────────

METADATA_COLUMNS = ["age_bucket", "audio_duration_sec", "utterance_id", "child_id"]


def tpu_transcribe(
    samples,
    base_model: str = "openai/whisper-medium.en",
    adapter_repo: str | None = None,
    adapter_subfolder: str = "",
    ref_column: str = "orthographic_text",
    max_new_tokens: int = 446,
    batch_size: int = 32,
    dtype: torch.dtype = torch.bfloat16,
    output_csv: str | None = None,
    hf_repo: str | None = None,
    hf_path_in_repo: str | None = None,
    hf_token: str | None = None,
) -> pd.DataFrame:
    """
    Run batched Whisper transcription across all available TPU chips.

    Args:
        samples:            List of dicts (e.g. from a streamed HF dataset).
                            Each dict must have an ``audio`` field with a
                            ``bytes`` sub-field, and the metadata columns
                            (age_bucket, audio_duration_sec, utterance_id,
                            child_id) plus ``ref_column``.
        base_model:         HF model name for the base Whisper checkpoint.
        adapter_repo:       HF repo for the LoRA adapter (None = no adapter).
        adapter_subfolder:  Subfolder inside adapter_repo.
        ref_column:         Column name for reference text.
        max_new_tokens:     Max tokens to generate per sample.
        batch_size:         Samples per chip per batch.
        dtype:              torch.float32 or torch.bfloat16.
        output_csv:         If set, save the DataFrame to this CSV path.
        hf_repo:            If set, upload the CSV to this HF repo.
        hf_path_in_repo:    Filename inside the HF repo (required if hf_repo).
        hf_token:           HF token for upload (uses cached login if None).

    Returns:
        pd.DataFrame with columns: pred, ref, age_bucket, duration,
        utterance_id, child_id.
    """
    assert len(samples) > 0, "No samples provided"

    # ── Pre-process audio ────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  TPU Whisper Inference: {base_model}")
    if adapter_repo:
        print(f"  Adapter: {adapter_repo}/{adapter_subfolder}")
    print(f"  {len(samples)} samples | batch_size={batch_size} | "
          f"max_new_tokens={max_new_tokens}")
    print(f"{'=' * 60}")

    tmpdir = tempfile.mkdtemp()

    print(f"  Pre-processing audio...")
    t0 = time.time()

    metadata = []
    for i, example in enumerate(samples):
        # Decode audio bytes -> float32 array @ 16 kHz
        audio_bytes = example["audio"]["bytes"]
        array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if sr != 16000:
            array = librosa.resample(array, orig_sr=sr, target_sr=16000)

        np.save(os.path.join(tmpdir, f"audio_{i}.npy"), array)

        metadata.append({
            "ref": example.get(ref_column, ""),
            "age_bucket": example.get("age_bucket", ""),
            "duration": example.get("audio_duration_sec", 0.0),
            "utterance_id": example.get("utterance_id", ""),
            "child_id": example.get("child_id", ""),
        })

        if (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{len(samples)} audio files decoded", flush=True)

    with open(os.path.join(tmpdir, "meta.json"), "w") as f:
        json.dump({"total": len(samples)}, f)

    print(f"  Audio pre-processed in {time.time() - t0:.1f}s")

    # ── Spawn workers ────────────────────────────────────
    dtype_str = str(dtype).split(".")[-1]
    t_total = time.time()

    try:
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(
            _transcribe_worker,
            args=(tmpdir, base_model, adapter_repo, adapter_subfolder,
                  max_new_tokens, batch_size, dtype_str),
        )
    except Exception as e:
        print(f"  xmp.spawn failed ({e}), trying single chip...")
        _transcribe_worker(
            0, tmpdir, base_model, adapter_repo, adapter_subfolder,
            max_new_tokens, batch_size, dtype_str,
        )

    # ── Collect results ──────────────────────────────────
    all_results = {}
    for fname in sorted(os.listdir(tmpdir)):
        if fname.startswith("results_"):
            with open(os.path.join(tmpdir, fname)) as f:
                all_results.update(json.load(f))

    shutil.rmtree(tmpdir)

    # ── Build DataFrame ──────────────────────────────────
    records = []
    for i in range(len(samples)):
        pred = all_results[str(i)]
        m = metadata[i]
        records.append({
            "pred": pred,
            "ref": m["ref"],
            "age_bucket": m["age_bucket"],
            "duration": m["duration"],
            "utterance_id": m["utterance_id"],
            "child_id": m["child_id"],
        })

    df = pd.DataFrame(records)

    elapsed = time.time() - t_total
    print(f"\n  Done! {len(df)} transcriptions in {elapsed:.1f}s")
    print(f"{'=' * 60}")

    # ── Save & upload ────────────────────────────────────
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"  Saved to {output_csv}")

    if hf_repo and hf_path_in_repo:
        from huggingface_hub import HfApi
        api = HfApi()
        csv_path = output_csv or "records.csv"
        if not output_csv:
            df.to_csv(csv_path, index=False)
        api.upload_file(
            path_or_fileobj=csv_path,
            path_in_repo=hf_path_in_repo,
            repo_id=hf_repo,
            repo_type="model",
            token=hf_token,
        )
        print(f"  Uploaded to {hf_repo}/{hf_path_in_repo}")
        if not output_csv:
            os.remove(csv_path)

    print()
    return df


# ── CLI entry point ──────────────────────────────────────────────

def main():
    import argparse
    from datasets import load_dataset, Audio
    from itertools import islice

    parser = argparse.ArgumentParser(
        description="TPU Whisper inference on a HuggingFace audio dataset",
    )

    # Model
    parser.add_argument("--base-model", default="openai/whisper-medium.en",
                        help="Base Whisper model name/path")
    parser.add_argument("--adapter-repo", default=None,
                        help="HF repo for LoRA adapter (omit for no adapter)")
    parser.add_argument("--adapter-subfolder", default="",
                        help="Subfolder inside adapter repo")

    # Dataset
    parser.add_argument("--dataset-repo", required=True,
                        help="HF dataset repo id")
    parser.add_argument("--dataset-split", default="validation",
                        help="Dataset split to use")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to process (None = all)")
    parser.add_argument("--ref-column", default="orthographic_text",
                        help="Column name for reference text")

    # Inference
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size per TPU chip")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Max tokens to generate per sample")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "bfloat16"],
                        help="Model dtype")

    # Output
    parser.add_argument("--output-csv", default="records.csv",
                        help="Path to save output CSV")
    parser.add_argument("--hf-repo", default=None,
                        help="HF repo to upload CSV to")
    parser.add_argument("--hf-path-in-repo", default=None,
                        help="Filename inside the HF repo")
    parser.add_argument("--hf-token", default=None,
                        help="HF token for upload (uses cached login if omitted)")

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset {args.dataset_repo} ({args.dataset_split})...")
    ds = load_dataset(args.dataset_repo, split=args.dataset_split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    if args.num_samples is not None:
        samples = list(islice(ds, args.num_samples))
    else:
        samples = list(ds)

    print(f"  {len(samples)} samples loaded")

    dtype = getattr(torch, args.dtype)

    df = tpu_transcribe(
        samples,
        base_model=args.base_model,
        adapter_repo=args.adapter_repo,
        adapter_subfolder=args.adapter_subfolder,
        ref_column=args.ref_column,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        dtype=dtype,
        output_csv=args.output_csv,
        hf_repo=args.hf_repo,
        hf_path_in_repo=args.hf_path_in_repo,
        hf_token=args.hf_token,
    )

    print(f"Results shape: {df.shape}")
    print(df.head())


if __name__ == "__main__":
    main()
