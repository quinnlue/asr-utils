import torch
from transformers import Seq2SeqTrainer, TrainerCallback
import os
import wandb
from score import score_wer

class TokenErrorRateTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer that also logs next-token error rate every logging step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._token_error_rates = []

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, **kwargs
        )
        with torch.no_grad():
            preds = outputs.logits.argmax(dim=-1)
            labels = inputs["labels"]
            mask = labels != -100
            if mask.any():
                ter = 1.0 - (preds[mask] == labels[mask]).float().mean().item()
            else:
                ter = 0.0
            self._token_error_rates.append(ter)
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if self._token_error_rates:
            logs["token_error_rate"] = (
                sum(self._token_error_rates) / len(self._token_error_rates)
            )
            self._token_error_rates = []
        super().log(logs, *args, **kwargs)


class PeriodicWERCallback(TrainerCallback):
    """Every N optimizer steps, run generate() on a small eval batch and log WER."""

    def __init__(self, eval_dataset, collate_fn, processor,
                 every_n_steps=100, num_samples=64):
        self.collate_fn = collate_fn
        self.processor = processor
        self.every_n_steps = every_n_steps

        # Cache raw samples (direct indexing — not streaming)
        num_samples = min(num_samples, len(eval_dataset))
        self._cached_samples = [eval_dataset[i] for i in range(num_samples)]
        print(f"Cached {len(self._cached_samples)} eval samples for periodic WER.")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step != 1 and (state.global_step % self.every_n_steps != 0 or state.global_step == 0):
            return

        n = len(self._cached_samples)
        batch = self.collate_fn(self._cached_samples)

        device = next(model.parameters()).device
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)

        model.eval()
        with torch.no_grad():
            gen_ids = model.generate(input_features=input_features, max_new_tokens=446)
        model.train()

        pred_str = self.processor.tokenizer.batch_decode(
            gen_ids, skip_special_tokens=True
        )
        lbl = labels.clone()
        lbl[lbl == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.tokenizer.batch_decode(
            lbl, skip_special_tokens=True
        )

        wer = score_wer(actual=label_str, predicted=pred_str)
        if wandb.run is not None:
            wandb.log({"batch_wer": wer}, step=state.global_step)
        print(f"\n[Step {state.global_step}] Batch WER ({n} samples): {wer:.4f}")


class AdapterSnapshotCallback(TrainerCallback):
    """Push every full checkpoint (with optimizer/scheduler state for resumption)
    AND the adapter-only snapshot to unique Hub subfolders on every save."""

    def __init__(self, hub_repo_id="whisper-small-finetune", local_dir="./adapters"):
        self.hub_repo_id = hub_repo_id
        self.local_dir = local_dir

    def on_save(self, args, state, control, model=None, **kwargs):
        from huggingface_hub import HfApi
        step = state.global_step
        api = HfApi()
        api.create_repo(repo_id=self.hub_repo_id, exist_ok=True)

        # --- 1. Upload full checkpoint (optimizer, scheduler, rng — everything for resume) ---
        ckpt_name = f"checkpoint-{step}"
        ckpt_path = os.path.join(args.output_dir, ckpt_name)
        if os.path.isdir(ckpt_path):
            try:
                api.upload_folder(
                    repo_id=self.hub_repo_id,
                    folder_path=ckpt_path,
                    path_in_repo=ckpt_name,
                    commit_message=f"Full checkpoint: {ckpt_name}",
                )
                print(f"\n[Step {step}] Full checkpoint pushed to Hub: {self.hub_repo_id}/{ckpt_name}")
            except Exception as e:
                print(f"\n[Step {step}] Failed to push full checkpoint: {e}")
        else:
            print(f"\n[Step {step}] WARNING: checkpoint dir not found at {ckpt_path}")

        # --- 2. Upload adapter-only snapshot (lightweight, for inference) ---
        adapter_name = f"adapter-step-{step}"
        adapter_path = os.path.join(self.local_dir, adapter_name)
        model.save_pretrained(adapter_path)
        print(f"[Step {step}] Adapter saved to {adapter_path}")

        try:
            api.upload_folder(
                repo_id=self.hub_repo_id,
                folder_path=adapter_path,
                path_in_repo=adapter_name,
                commit_message=f"Adapter snapshot: {adapter_name}",
            )
            print(f"[Step {step}] Adapter pushed to Hub: {self.hub_repo_id}/{adapter_name}")
        except Exception as e:
            print(f"[Step {step}] Failed to push adapter: {e}")


print("Custom trainer, WER callback, and adapter snapshot callback defined.")