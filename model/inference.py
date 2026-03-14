from score import score_wer
from datasets import load_dataset
import torch
import librosa
import pandas as pd
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from peft import PeftModel
from datasets import load_dataset, Audio
from itertools import islice
import random
import io
import soundfile as sf

BASE_MODEL = "openai/whisper-medium.en"
ADAPTER_REPO = "quinnlue/whisper-medium-ckpts"
ADAPTER_SUBFOLDER = ""

DATASET_REPO = "quinnlue/audio-cleaned"
DATASET_SPLIT = "validation"
FIRST_N_SAMPLES = 15700

def get_model():
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(BASE_MODEL)

    # Load LoRA adapter from Hugging Face Hub
    model = PeftModel.from_pretrained(base_model, ADAPTER_REPO, subfolder=ADAPTER_SUBFOLDER)
    model = model.merge_and_unload()  # merge for faster inference

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    print(f"Model loaded on {device} with adapter from {ADAPTER_REPO}/{ADAPTER_SUBFOLDER}")

    return model, processor


def get_data():
    ds = load_dataset(
        DATASET_REPO,
        split=DATASET_SPLIT,
        streaming=True,
    )
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = list(islice(ds, FIRST_N_SAMPLES))
    return ds


def transcribe(model, processor, audio_path: str) -> str:
    """Transcribe an audio file using the fine-tuned Whisper model."""
    array, sr = sf.read(audio_path, dtype="float32")
    if sr != 16000:
        array = librosa.resample(array, orig_sr=sr, target_sr=16000)

    inputs = processor.feature_extractor(
        array, sampling_rate=16000, return_tensors="pt"
    )
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        generated_ids = model.generate(input_features=input_features, max_new_tokens=446)

    transcription = processor.tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()
    return transcription

# Example usage (replace with your audio file path):
# print(transcribe("path/to/audio.wav"))