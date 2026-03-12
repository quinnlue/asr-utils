#!/usr/bin/env python3
"""whisper-small — data parallel across chips + batched per chip"""

import torch
import time
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.xla_multiprocessing as xmp
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_NAME     = "openai/whisper-small"
MAX_NEW_TOKENS = 100        # max decoder tokens to generate
DTYPE          = torch.float32
BATCH_PER_CHIP = 2          # adjust to fit TPU memory


def generate_batched(model, processor, audio_features, device, max_new_tokens=MAX_NEW_TOKENS):
    """
    Generate transcriptions for a batch of audio features on a single TPU device.
    """
    bsz = audio_features.shape[0]

    # 1️⃣ Encode audio once per batch
    encoder_outputs = model.get_encoder()(audio_features)
    xm.mark_step()

    # 2️⃣ Initialize decoder input IDs with pad token
    decoder_input_ids = torch.full(
        (bsz, 1),
        processor.tokenizer.pad_token_id,
        dtype=torch.long,
        device=device
    )

    generated_tokens = [decoder_input_ids]

    # 3️⃣ Autoregressive decoding loop
    for i in range(max_new_tokens):
        outputs = model(
            input_features=None,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            use_cache=True,
        )
        next_token = outputs.logits[:, -1:].argmax(dim=-1)
        decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
        generated_tokens.append(next_token)

        if i % 5 == 0:
            xm.mark_step()

    xm.mark_step()
    return decoder_input_ids[:, 1:]  # remove initial pad token


def _worker_with_features(index, all_features):
    """Worker receives pre-processed input_features tensor (batch, n_mels, 3000)."""
    device = torch_xla.device()
    world_size = xr.world_size()

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=DTYPE)
    model.eval().to(device)
    xm.mark_step()
    xm.wait_device_ops()

    # Split pre-processed features across chips
    my_indices = list(range(index, all_features.shape[0], world_size))
    input_features = all_features[my_indices].to(device)

    # Generation
    output_ids = generate_batched(model, processor, input_features, device)

    # Decode
    for i, idx in enumerate(my_indices):
        transcription = processor.tokenizer.decode(
            output_ids[i].cpu().tolist(),
            skip_special_tokens=True
        )
        print(f"[Chip {index} | audio {idx}] {transcription}")

def _worker(index):
    device = torch_xla.device()
    world_size = xr.world_size()

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=DTYPE)
    model.eval().to(device)
    xm.mark_step()
    xm.wait_device_ops()

    # Example audio "prompts" — replace with your actual audio tensors
    all_audios = [
        torch.randn(16000 * 5) for _ in range(8)  # 5s random audio, 16kHz
    ]

    # Split across chips
    my_audios = [a for i, a in enumerate(all_audios) if i % world_size == index]

    if index == 0:
        print(f"\n🔢 {world_size} chips | {BATCH_PER_CHIP} audios/chip batched")
        print(f"📝 {len(all_audios)} total audio files\n")

    # Process audio to input features (batched)
    inputs = processor(my_audios, sampling_rate=16000, return_tensors="pt", padding=True)
    input_features = inputs.input_features.to(device)

    # Warmup
    if index == 0:
        print("⏳ Warmup...")
    t0 = time.time()
    _ = generate_batched(model, processor, input_features, device)
    xm.wait_device_ops()
    if index == 0:
        print(f"✓ Compiled in {time.time() - t0:.1f}s\n")

    # Timed generation
    if index == 0:
        print("🚀 Generating...\n")
    xm.wait_device_ops()
    t0 = time.time()

    output_ids = generate_batched(model, processor, input_features, device)
    xm.mark_step()
    xm.wait_device_ops()
    elapsed = time.time() - t0

    # Decode outputs
    for i, audio in enumerate(my_audios):
        transcription = processor.tokenizer.decode(
            output_ids[i].cpu().tolist(),
            skip_special_tokens=True
        )
        print(f"\n[Chip {index} | audio {i}] {'─' * 40}")
        print(transcription)

    if index == 0:
        total_tokens = len(all_audios) * MAX_NEW_TOKENS
        print(f"\n{'═' * 60}")
        print(f"📊 {len(all_audios)} audio files × {MAX_NEW_TOKENS} tokens = {total_tokens}")
        print(f"⏱  Wall time: {elapsed:.2f}s")
        print(f"🚀 Aggregate: ~{total_tokens / elapsed:.0f} tok/s")


if __name__ == "__main__":
    import librosa

    raw_audios = [
        librosa.load(librosa.ex(name), sr=16000)[0]
        for name in ["libri2", "libri2", "libri2", "libri2"]
    ]

    # Preprocess on CPU — pads mel features to 3000 automatically
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    inputs = processor(raw_audios, sampling_rate=16000, return_tensors="pt", padding=True)
    input_features = inputs.input_features  # (batch, n_mels, 3000)

    # Pass pre-processed features to workers
    xmp.spawn(_worker_with_features, args=(input_features,))