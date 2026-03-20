import argparse
import io
import json
import os
import re
import tempfile
import time
from datetime import datetime

import jax.numpy as jnp
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForSpeechSeq2Seq, WhisperConfig, WhisperProcessor
from transformers.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax
from whisper_jax.modeling_flax_whisper import FlaxWhisperForConditionalGeneration
from whisper_jax.pipeline import FlaxWhisperPipline

from utils.score import score_wer


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run Whisper inference with whisper_jax",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Data")
    g.add_argument("--dataset", default="quinnlue/audio-cleaned-test", help="HF dataset ID")
    g.add_argument("--split", default="test", help="Dataset split to run inference on")
    g.add_argument("--first-n", type=int, default=None, help="Only use the first N samples from the dataset")

    g = p.add_argument_group("Model")
    g.add_argument("--model", default="openai/whisper-medium.en", help="Pretrained Whisper model ID")
    g.add_argument("--adapter", default="quinnlue/whisper-med-mlp", help="Optional LoRA adapter path or HF ID")
    g.add_argument("--adapter-subfolder", default="final", help="Subfolder within the adapter repo")

    g = p.add_argument_group("Generation")
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Retained for CLI compatibility; whisper_jax uses the model generation config.",
    )
    g.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="Retained for CLI compatibility; unused by whisper_jax inference.",
    )
    g.add_argument("--num-beams", type=int, default=1, help="Number of beams to use for beam search")

    g = p.add_argument_group("Precision")
    g.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True, help="Use bfloat16")

    g = p.add_argument_group("Output")
    g.add_argument("--output-csv", default=None, help="Path to write CSV results")
    g.add_argument("--output-jsonl", default=None, help="Path to write JSONL results")
    g.add_argument("--no-stdout", action="store_true", default=False, help="Suppress printing transcriptions to stdout")

    g = p.add_argument_group("HF Upload")
    g.add_argument(
        "--upload-to-hf",
        default="quinnlue/asr-evals",
        metavar="REPO_ID",
        help="HF Hub repo ID to upload inference CSV to (e.g. 'myuser/asr-results'). Repo is created if it doesn't exist.",
    )
    g.add_argument(
        "--upload-filename",
        default=None,
        help="Filename for the uploaded CSV in the HF repo. Defaults to '<dataset>_<adapter|model>_<timestamp>.csv'",
    )

    return p.parse_args(argv)


_METADATA_COLS = (
    "child_id",
    "session_id",
    "audio_duration_sec",
    "age_bucket",
    "md5_hash",
    "filesize_bytes",
)


def collate_fn(batch):
    """Decode raw audio bytes and resample to 16 kHz."""
    waveforms = []
    texts = []
    utterance_ids = []
    metadata = {col: [] for col in _METADATA_COLS}

    for sample in batch:
        try:
            waveform, sr = sf.read(io.BytesIO(sample["audio"]["bytes"]), dtype="float32")
        except Exception as exc:
            print(f"[collate_fn] Skipping broken audio (path={sample['audio'].get('path', '?')}): {exc}")
            continue

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if len(waveform) == 0:
            print(f"[collate_fn] Skipping empty audio (path={sample['audio'].get('path', '?')})")
            continue
        if sr != 16_000:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16_000)

        waveforms.append(np.asarray(waveform, dtype=np.float32))
        texts.append(sample.get("orthographic_text", ""))
        utterance_ids.append(sample.get("utterance_id", ""))
        for col in _METADATA_COLS:
            metadata[col].append(sample.get(col, ""))

    if not waveforms:
        print("[collate_fn] WARNING: entire batch was broken - returning dummy batch")
        waveforms = [np.zeros(16_000, dtype=np.float32)]
        texts = [""]
        utterance_ids = [""]
        for col in _METADATA_COLS:
            metadata[col].append("")

    return {
        "waveforms": waveforms,
        "texts": texts,
        "utterance_ids": utterance_ids,
        "metadata": metadata,
    }


def get_pipeline(args):
    dtype = jnp.bfloat16 if args.bf16 else jnp.float32

    if getattr(args, "merged_model_path", None):
        base_checkpoint = args.original_model
        print(f"Getting pipeline for {base_checkpoint}")
        processor = WhisperProcessor.from_pretrained(base_checkpoint)
        config = WhisperConfig.from_pretrained(base_checkpoint)
        model = FlaxWhisperForConditionalGeneration(config, dtype=dtype)

        print(f"Loading merged model from {args.merged_model_path}")
        pt_state = torch.load(args.merged_model_path, map_location="cpu")
        params = convert_pytorch_state_dict_to_flax(pt_state, model)

        print("Creating pipeline")
        return FlaxWhisperPipline(
            checkpoint=base_checkpoint,
            model=(model, params),
            processor=processor,
            tokenizer=processor.tokenizer,
            dtype=dtype,
            batch_size=args.batch_size,
        )

    print(f"Getting pipeline for {args.model}")
    return FlaxWhisperPipline(
        checkpoint=args.model,
        dtype=dtype,
        batch_size=args.batch_size,
    )


def _iter_beam_outputs(output):
    if "beams" in output and output["beams"]:
        return output["beams"]
    return [output]


def _build_result_rows(outputs, utt_ids, texts, batch_meta, has_ground_truth, expand_beams):
    rows = []
    top_predictions = []

    for idx, output in enumerate(outputs):
        top_predictions.append(output.get("text", ""))
        beam_outputs = _iter_beam_outputs(output) if expand_beams else [output]

        for beam_rank, beam_output in enumerate(beam_outputs):
            row = {
                "utterance_id": utt_ids[idx],
                "prediction": beam_output.get("text", ""),
            }
            if has_ground_truth:
                row["reference"] = texts[idx]
            for col in _METADATA_COLS:
                row[col] = batch_meta[col][idx]
            if expand_beams:
                row["beam_rank"] = beam_rank
                row["is_top_beam"] = beam_rank == 0
            rows.append(row)

    return rows, top_predictions


def _print_predictions(rows, expand_beams):
    for row in rows:
        if expand_beams:
            print(f"  [{row['utterance_id']}] beam={row['beam_rank']} {row['prediction']}")
        else:
            print(f"  [{row['utterance_id']}] {row['prediction']}")


def _default_hf_filename(args):
    def _slug(value):
        return re.sub(r"[-_/]+", "-", value.rsplit("/", 1)[-1]).strip("-")

    ds_slug = _slug(args.dataset)
    model_slug = _slug(args.original_model)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.original_adapter:
        adapter_slug = _slug(args.original_adapter)
        return f"{ds_slug}_{model_slug}_{adapter_slug}_{ts}.csv"
    return f"{ds_slug}_{model_slug}_{ts}.csv"


def main(args):
    tmp_dir = tempfile.mkdtemp(prefix="whisper_infer_")
    args.original_model = args.model
    args.original_adapter = args.adapter
    args.merged_model_path = None

    if args.adapter is not None:
        model_dtype = torch.bfloat16 if args.bf16 else torch.float32
        print(f"Pre-merging LoRA adapter {args.adapter!r} into {args.model!r} on CPU ...")
        base = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, torch_dtype=model_dtype)
        subfolder_kwargs = {"subfolder": args.adapter_subfolder} if args.adapter_subfolder else {}
        base = PeftModel.from_pretrained(base, args.adapter, **subfolder_kwargs)
        base = base.merge_and_unload()

        merged_dir = os.path.join(tmp_dir, "merged_model")
        base.save_pretrained(merged_dir, safe_serialization=False)
        del base

        args.merged_model_path = os.path.join(merged_dir, "pytorch_model.bin")
        print(f"Pre-merge complete - saved merged checkpoint to {args.merged_model_path}")

    print(f"Loading dataset {args.dataset!r} split={args.split!r} ...")
    ds = load_dataset(args.dataset, split=args.split)
    ds = ds.cast_column("audio", Audio(decode=False))
    if args.first_n is not None:
        ds = ds.select(range(min(args.first_n, len(ds))))

    has_ground_truth = "orthographic_text" in ds.column_names
    dataloader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    pipeline = get_pipeline(args)
    expand_beams = args.num_beams > 1
    merged = []
    top_predictions = []
    all_refs = []
    num_examples = 0
    total_batches = (len(ds) + args.batch_size - 1) // args.batch_size
    t_start = time.time()

    print(f"Running inference ({total_batches} batches, batch_size={args.batch_size}) ...")
    for batch_idx, batch_data in enumerate(dataloader):
        waveforms = batch_data["waveforms"]
        texts = batch_data["texts"]
        utt_ids = batch_data["utterance_ids"]
        batch_meta = batch_data["metadata"]

        outputs = pipeline(
            waveforms,
            batch_size=args.batch_size,
            task="transcribe",
            num_beams=args.num_beams,
        )
        if not isinstance(outputs, list):
            outputs = [outputs]

        batch_rows, batch_top_predictions = _build_result_rows(
            outputs=outputs,
            utt_ids=utt_ids,
            texts=texts,
            batch_meta=batch_meta,
            has_ground_truth=has_ground_truth,
            expand_beams=expand_beams,
        )
        merged.extend(batch_rows)
        top_predictions.extend(batch_top_predictions)
        if has_ground_truth:
            all_refs.extend(texts)
        num_examples += len(utt_ids)

        if not args.no_stdout:
            _print_predictions(batch_rows, expand_beams=expand_beams)

        elapsed = time.time() - t_start
        print(f"  batch {batch_idx + 1}/{total_batches} ({elapsed:.1f}s elapsed)")

    if has_ground_truth:
        wer = score_wer(actual=all_refs, predicted=top_predictions)
        print(f"\nWER: {wer:.4f}")
    else:
        wer = None
        print("\nNo ground-truth text - WER not computed.")

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        pd.DataFrame(merged).to_csv(args.output_csv, index=False)
        print(f"Saved CSV -> {args.output_csv}")

    if args.output_jsonl:
        os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
        with open(args.output_jsonl, "w", encoding="utf-8") as f:
            for row in merged:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved JSONL -> {args.output_jsonl}")

    if args.upload_to_hf:
        from huggingface_hub import HfApi

        hf_filename = args.upload_filename or _default_hf_filename(args)
        csv_path = os.path.join(tmp_dir, hf_filename)
        pd.DataFrame(merged).to_csv(csv_path, index=False)

        api = HfApi()
        api.create_repo(args.upload_to_hf, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=csv_path,
            path_in_repo=hf_filename,
            repo_id=args.upload_to_hf,
            repo_type="dataset",
        )
        print(f"Uploaded to HF Hub -> {args.upload_to_hf}/{hf_filename}")

    elapsed = time.time() - t_start
    rate = num_examples / elapsed if elapsed else 0.0
    print(f"\nDone - {num_examples} samples in {elapsed:.1f}s ({rate:.1f} samples/s)")
    if wer is not None:
        print(f"WER: {wer:.4f}")


if __name__ == "__main__":
    main(parse_args())
