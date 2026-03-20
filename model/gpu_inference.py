import argparse
import io
import json
import os
import re
from datetime import datetime
import jiwer
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from utils.maps import english_spelling_normalizer
from utils.score import score_wer


_METADATA_COLS = (
    "child_id",
    "session_id",
    "audio_duration_sec",
    "age_bucket",
    "md5_hash",
    "filesize_bytes",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run single-GPU Whisper inference with optional merged PEFT adapter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("Data")
    data.add_argument("--dataset", default="quinnlue/audio-cleaned-test", help="HF dataset ID")
    data.add_argument("--split", default="test", help="Dataset split to run inference on")
    data.add_argument(
        "--first-n",
        type=int,
        default=None,
        help="Only use the first N samples from the dataset",
    )

    model = parser.add_argument_group("Model")
    model.add_argument("--model", default="openai/whisper-medium.en", help="Base Whisper model ID")
    model.add_argument(
        "--adapter",
        default="quinnlue/whisper-med-mlp",
        help="Optional LoRA adapter path or HF ID. Empty string disables adapter loading.",
    )
    model.add_argument(
        "--adapter-subfolder",
        default="final",
        help="Subfolder within the adapter repo. Empty string disables subfolder usage.",
    )

    generation = parser.add_argument_group("Generation")
    generation.add_argument("--batch-size", type=int, default=16, help="Inference batch size")
    generation.add_argument("--max-new-tokens", type=int, default=128, help="Maximum generated tokens")
    generation.add_argument("--num-beams", type=int, default=1, help="Number of beams to return")
    generation.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    generation.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Number of batches to prefetch per worker when num_workers > 0",
    )
    generation.add_argument(
        "--output-scores",
        action="store_true",
        default=False,
        help="Include generation sequence scores in the output rows.",
    )

    precision = parser.add_argument_group("Precision")
    precision.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True, help="Use bfloat16")

    output = parser.add_argument_group("Output")
    output.add_argument(
        "--output-csv",
        nargs="?",
        const="",
        default="",
        help="Path to write CSV results. If omitted, writes a descriptive default CSV filename.",
    )
    output.add_argument("--output-jsonl", default=None, help="Path to write JSONL results")
    output.add_argument(
        "--stdout",
        action="store_true",
        default=False,
        help="Print top beam predictions to stdout.",
    )

    return parser.parse_args(argv)


def ensure_single_gpu() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gpu_inference.py, but no GPU is available.")
    return torch.device("cuda:0")


def resolve_dtype(use_bf16: bool) -> torch.dtype:
    return torch.bfloat16 if use_bf16 else torch.float16


def load_processor_and_model(args, device: torch.device):
    dtype = resolve_dtype(args.bf16)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        attn_implementation="sdpa",
        torch_dtype=dtype,
    )

    if args.adapter:
        adapter_kwargs = {}
        if args.adapter_subfolder:
            adapter_kwargs["subfolder"] = args.adapter_subfolder
        model = PeftModel.from_pretrained(model, args.adapter, **adapter_kwargs)
        model = model.merge_and_unload()

    model.to(device)
    model.eval()
    return processor, model


def load_split(args):
    dataset = load_dataset(args.dataset, split=args.split)
    if "audio" in dataset.column_names:
        dataset = dataset.cast_column("audio", Audio(decode=False))
    if args.first_n is not None:
        dataset = dataset.select(range(min(args.first_n, len(dataset))))
    if "dataset_index" not in dataset.column_names:
        dataset = dataset.add_column("dataset_index", list(range(len(dataset))))
    return dataset


def build_collate_fn(processor):
    def collate_fn(batch):
        waveforms = []
        samples = []

        for sample in batch:
            waveform, _ = sf.read(io.BytesIO(sample["audio"]["bytes"]), dtype="float32")
            samples.append(sample)
            waveforms.append(np.asarray(waveform, dtype=np.float32))

        features = processor.feature_extractor(
            waveforms,
            sampling_rate=16_000,
            return_tensors="pt",
        ).input_features
        return {"samples": samples, "input_features": features}

    return collate_fn


def make_normalizer():
    return EnglishTextNormalizer(english_spelling_normalizer)


def compute_example_wer(reference: str | None, prediction: str, normalizer) -> float | None:
    if reference is None:
        return None

    norm_ref = normalizer(reference)
    norm_pred = normalizer(prediction)
    ref_words = len(norm_ref.split()) if norm_ref.strip() else 0
    if ref_words == 0:
        return 0.0
    return jiwer.wer(norm_ref, norm_pred)


def get_optional_value(sample, key):
    value = sample.get(key)
    if value is None:
        return None
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def generation_kwargs(args):
    return {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_beams,
        "return_dict_in_generate": True,
        "output_scores": args.output_scores,
    }


def decode_batch(processor, generated_sequences):
    return processor.tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "unknown"


def default_csv_path(args) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = slugify(args.adapter or args.model)
    split_name = slugify(args.split)
    return f"{split_name}_{model_name}_{timestamp}.csv"


def build_result_row(sample, beam_rank: int, prediction: str, sequence_score, normalizer):
    reference = sample.get("orthographic_text")
    row = {
        "dataset_index": sample.get("dataset_index"),
        "utterance_id": sample.get("utterance_id"),
        "beam_rank": beam_rank,
        "prediction": prediction,
        "sequence_score": sequence_score,
        "reference": reference,
        "wer": compute_example_wer(reference, prediction, normalizer),
    }
    for col in _METADATA_COLS:
        if col in sample:
            row[col] = get_optional_value(sample, col)
    return row


def run_inference(args):
    device = ensure_single_gpu()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    processor, model = load_processor_and_model(args, device)
    dataset = load_split(args)
    normalizer = make_normalizer()
    collate_fn = build_collate_fn(processor)
    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "collate_fn": collate_fn,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    rows = []
    top_predictions = []
    top_references = []
    progress = tqdm(dataloader, total=len(dataloader), desc="Generating", unit="batch")

    with torch.inference_mode():
        for batch in progress:
            if batch is None:
                continue

            input_features = batch["input_features"].to(device=device, dtype=model.dtype, non_blocking=True)
            outputs = model.generate(input_features=input_features, **generation_kwargs(args))
            predictions = decode_batch(processor, outputs.sequences)
            sequence_scores = getattr(outputs, "sequences_scores", None)

            if sequence_scores is not None:
                sequence_scores = sequence_scores.detach().float().cpu().tolist()
            else:
                sequence_scores = [None] * len(predictions)

            batch_samples = batch["samples"]
            for batch_index, sample in enumerate(batch_samples):
                reference = sample.get("orthographic_text")
                base_idx = batch_index * args.num_beams
                for beam_rank in range(args.num_beams):
                    pred_idx = base_idx + beam_rank
                    prediction = predictions[pred_idx]
                    row = build_result_row(
                        sample=sample,
                        beam_rank=beam_rank,
                        prediction=prediction,
                        sequence_score=sequence_scores[pred_idx],
                        normalizer=normalizer,
                    )
                    rows.append(row)

                    if beam_rank == 0:
                        top_predictions.append(prediction)
                        if reference is not None:
                            top_references.append(reference)
                        if args.stdout:
                            sample_id = sample.get("utterance_id", sample.get("dataset_index"))
                            print(f"[{sample_id}] {prediction}")

    result_df = pd.DataFrame(rows)
    aggregate_wer = None
    if top_references and len(top_references) == len(top_predictions):
        aggregate_wer = score_wer(actual=top_references, predicted=top_predictions)

    return result_df, aggregate_wer


def write_outputs(df: pd.DataFrame, csv_path: str | None, jsonl_path: str | None):
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Wrote CSV results to {csv_path}")

    if jsonl_path:
        os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
        with open(jsonl_path, "w", encoding="utf-8") as handle:
            for row in df.to_dict(orient="records"):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote JSONL results to {jsonl_path}")


def main(argv=None):
    args = parse_args(argv)
    results_df, aggregate_wer = run_inference(args)
    csv_path = args.output_csv if args.output_csv else default_csv_path(args)
    write_outputs(results_df, csv_path, args.output_jsonl)

    print(f"Generated {len(results_df)} rows.")
    if aggregate_wer is None:
        print("Aggregate top-1 WER: n/a (dataset split has no orthographic_text labels)")
    else:
        print(f"Aggregate top-1 WER: {aggregate_wer:.4f}")


if __name__ == "__main__":
    main()
