import torch
from transformers import StaticCache
from transformers.cache_utils import EncoderDecoderCache, StaticCache, DynamicCache
import torch
import torch_xla.core.xla_model as xm
from transformers import StaticCache, DynamicCache

@torch.no_grad()
def generate(
    model,
    processor,
    batch,
    max_new_tokens,
    device,
    max_seq_len,
):
    bsz = batch.shape[0]
    gc = model.generation_config

    # ── Full 4-token prefix (matching HF generate) ──
    start       = model.config.decoder_start_token_id              # <|startoftranscript|>
    lang_token  = processor.tokenizer.convert_tokens_to_ids("<|en|>")
    task_token  = processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
    notimestamp = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    prefix = [start, lang_token, task_token, notimestamp]

    # ── Suppression lists from generation_config ──
    suppress_ids = torch.tensor(gc.suppress_tokens, device=device)
    begin_suppress_ids = torch.tensor(gc.begin_suppress_tokens, device=device)

    decoder_input_ids = torch.tensor([prefix] * bsz, device=device)  # (bsz, 4)
    bsz, prompt_len = decoder_input_ids.shape

    cache = EncoderDecoderCache(
        StaticCache(
            config=model.config,
            max_batch_size=bsz,
            max_cache_len=max_seq_len,
            device=device,
            dtype=model.dtype,
        ),
        DynamicCache(),
    )

    features = processor(batch, sampling_rate=16000, return_tensors="pt").input_features.to(device=device, dtype=model.dtype)
    encoder_hidden_states = model.model.encoder(features).last_hidden_state

    # ── Prefill ──
    cache_pos = torch.arange(prompt_len, device=device)
    decoder_out = model.model.decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=encoder_hidden_states,
        past_key_values=cache,
        cache_position=cache_pos,
        use_cache=True,
        output_hidden_states=True,
    )
    logits = model.proj_out(decoder_out.last_hidden_state)

    # Apply suppression to first generated token
    logits[:, -1, suppress_ids] = float("-inf")
    logits[:, -1, begin_suppress_ids] = float("-inf")

    eos_token_id = model.generation_config.eos_token_id
    finished = torch.zeros(bsz, dtype=torch.bool, device=device)

    next_token = logits[:, -1:].argmax(dim=-1)
    # Mark sequences that already hit EOS
    finished = finished | (next_token.squeeze(-1) == eos_token_id)
    generated = [next_token]
    all_hidden_states = [decoder_out.hidden_states]
    xm.mark_step()

    # ── Decode loop ──
    for i in range(1, max_new_tokens):
        # Force finished sequences to feed EOS back into the decoder
        # so they don't produce new (repeated) content
        next_token = torch.where(
            finished.unsqueeze(-1),
            torch.tensor([[eos_token_id]], device=device),
            next_token,
        )

        cache_pos = torch.tensor(
            [prompt_len + i - 1], device=device, dtype=torch.long
        )
        decoder_out = model.model.decoder(
            input_ids=next_token,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=cache,
            cache_position=cache_pos,
            use_cache=True,
            output_hidden_states=True,
        )
        logits = model.proj_out(decoder_out.last_hidden_state)

        # Apply suppression at every step
        logits[:, -1, suppress_ids] = float("-inf")

        next_token = logits[:, -1:].argmax(dim=-1)
        # Override with EOS for already-finished sequences
        next_token = torch.where(
            finished.unsqueeze(-1),
            torch.tensor([[eos_token_id]], device=device),
            next_token,
        )
        # Update finished mask
        finished = finished | (next_token.squeeze(-1) == eos_token_id)

        generated.append(next_token)
        all_hidden_states.append(decoder_out.hidden_states)
        xm.mark_step()

    return torch.cat(generated, dim=-1)