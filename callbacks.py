import time

import torch
from transformers import Seq2SeqTrainer, TrainerCallback
import os
import wandb
from score import score_wer

class TokenErrorRateTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer that also logs next-token error rate and
    dataloader-wait profiling every logging step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._token_error_rates = []
        # dataloader profiling accumulators
        self._dataloader_times: list[float] = []
        self._training_step_times: list[float] = []

    def _load_optimizer_and_scheduler(self, checkpoint):
        """Load optimizer & scheduler from *checkpoint*, but on mismatch
        print a detailed per-group diff before re-raising."""
        try:
            super()._load_optimizer_and_scheduler(checkpoint)
        except (ValueError, RuntimeError) as e:
            # ── build diagnostic ──
            ckpt_path = os.path.join(checkpoint, "optimizer.pt")
            lines = [
                "",
                "=" * 72,
                "OPTIMIZER STATE MISMATCH — cannot resume optimizer",
                "=" * 72,
                f"Checkpoint: {ckpt_path}",
                f"Error:      {e}",
                "",
            ]

            try:
                saved = torch.load(ckpt_path, map_location="cpu",
                                   weights_only=True)
                saved_groups = saved.get("param_groups", [])
            except Exception:
                saved_groups = None

            if saved_groups is not None:
                cur_groups = self.optimizer.param_groups
                lines.append(
                    f"  # param groups  —  checkpoint: {len(saved_groups)}"
                    f"  vs  current: {len(cur_groups)}"
                )
                for i in range(max(len(saved_groups), len(cur_groups))):
                    s_n = len(saved_groups[i]["params"]) if i < len(saved_groups) else "—"
                    c_n = len(cur_groups[i]["params"])    if i < len(cur_groups)   else "—"
                    flag = " ✓" if s_n == c_n else " ✗ MISMATCH"
                    lines.append(
                        f"  group {i}:  checkpoint params={s_n}"
                        f"  vs  current params={c_n}{flag}"
                    )

                # Show what the current groups actually contain
                lines.append("")
                lines.append("Current optimizer parameter groups:")
                for i, g in enumerate(cur_groups):
                    param_names = []
                    opt_param_ids = {id(p) for p in g["params"]}
                    for name, p in self.model.named_parameters():
                        if id(p) in opt_param_ids:
                            param_names.append(name)
                    lines.append(
                        f"  group {i} ({len(g['params'])} params, "
                        f"lr={g.get('lr')}, wd={g.get('weight_decay')}):"
                    )
                    for pn in param_names[:20]:
                        lines.append(f"    {pn}")
                    if len(param_names) > 20:
                        lines.append(f"    ... and {len(param_names) - 20} more")

            lines.append("=" * 72)
            print("\n".join(lines))
            raise

    # ── profiling hooks ───────────────────────────────────────────

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        """Time spent here = CPU blocked waiting for the dataloader."""
        t0 = time.perf_counter()
        result = super().get_batch_samples(epoch_iterator, num_batches, device)
        self._dataloader_times.append(time.perf_counter() - t0)
        return result

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Time spent here = forward + backward + (sync if DDP)."""
        t0 = time.perf_counter()
        loss = super().training_step(model, inputs, num_items_in_batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._training_step_times.append(time.perf_counter() - t0)
        return loss

    # ── existing hooks ────────────────────────────────────────────

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

        # ── dataloader profiling stats ──
        if self._dataloader_times:
            total_dl   = sum(self._dataloader_times)
            total_step = sum(self._training_step_times)
            n_dl       = len(self._dataloader_times)
            n_step     = len(self._training_step_times)
            total      = total_dl + total_step

            logs["dataloader_s_per_batch"]  = total_dl / n_dl
            logs["step_s_per_batch"]        = total_step / n_step if n_step else 0.0
            logs["dataloader_wait_pct"]     = (total_dl / total * 100) if total > 0 else 0.0

            self._dataloader_times = []
            self._training_step_times = []

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
            gen_ids = model.generate(input_features=input_features, max_new_tokens=128)
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


print("Custom trainer and WER callback defined.")