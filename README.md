# Whisper LoRA Fine-Tuning

Fine-tune a Whisper model with LoRA adapters and waveform/mel-level data augmentations.

## Quick start

```bash
# Train with defaults
python train.py

# Resume from a Hub checkpoint
python train.py --resume-from last-checkpoint

# Custom run
python train.py --model openai/whisper-small.en --batch-size 32 --epochs 5 --lr 3e-4
```

All flags have sensible defaults — running with no arguments reproduces the baseline config.

## CLI reference

### Data

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `quinnlue/audio` | HF dataset ID (must have train/validation/test splits) |
| `--noise-dataset` | `quinnlue/realclass` | HF dataset ID for noise samples used in augmentation |
| `--val-size` | `2400` | Number of validation samples to use (`0` = all) |

### Model

| Flag | Default | Description |
|---|---|---|
| `--model` | `openai/whisper-medium.en` | Pretrained Whisper checkpoint |

### LoRA

| Flag | Default | Description |
|---|---|---|
| `--lora-r` | `32` | LoRA rank |
| `--lora-alpha` | `32` | LoRA alpha scaling factor |
| `--lora-dropout` | `0.0` | LoRA dropout |
| `--lora-target-modules` | `q_proj v_proj k_proj out_proj fc1 fc2` | Modules to apply LoRA to |

### Training

| Flag | Default | Description |
|---|---|---|
| `--batch-size` | `48` | Per-device train batch size |
| `--grad-accum` | `2` | Gradient accumulation steps |
| `--epochs` | `3` | Number of training epochs |
| `--lr` | `5e-4` | Peak learning rate |
| `--lr-scheduler` | `cosine` | LR scheduler type |
| `--warmup-steps` | `500` | Linear warmup steps |
| `--weight-decay` | `0.01` | Weight decay |
| `--adam-beta1` | `0.9` | Adam β₁ |
| `--adam-beta2` | `0.98` | Adam β₂ |
| `--optim` | `adamw_torch` | Optimizer name |
| `--bf16 / --no-bf16` | `True` | bf16 mixed precision |
| `--gradient-checkpointing / --no-gradient-checkpointing` | `False` | Activation checkpointing |
| `--torch-compile / --no-torch-compile` | `True` | `torch.compile` via Trainer |

### Evaluation & checkpointing

| Flag | Default | Description |
|---|---|---|
| `--eval-steps` | `500` | Evaluate every N steps |
| `--save-steps` | `500` | Save checkpoint every N steps |
| `--save-total-limit` | `32` | Max checkpoints kept on disk |
| `--logging-steps` | `25` | Log metrics every N steps |
| `--generation-max-length` | `446` | Max tokens for `generate()` during eval |

### Output & Hub

| Flag | Default | Description |
|---|---|---|
| `--output-dir` | `./whisper-medium-finetune` | Local checkpoint directory |
| `--hub-model-id` | `quinnlue/whisper-medium-finetune` | HF Hub repo ID |
| `--push-to-hub / --no-push-to-hub` | `True` | Push checkpoints to Hub |
| `--hub-strategy` | `checkpoint` | When to push (`checkpoint`, `end`, `every_save`, `all_checkpoints`) |
| `--report-to` | `wandb` | Experiment tracker (`wandb`, `tensorboard`, `none`) |

### Resumption

| Flag | Default | Description |
|---|---|---|
| `--resume-from` | *(none)* | Checkpoint name to download and resume from (e.g. `last-checkpoint`). Omit to train from scratch. |

### Callbacks

| Flag | Default | Description |
|---|---|---|
| `--wer-every-n-steps` | `100` | Run periodic WER evaluation every N steps |
| `--wer-num-samples` | `64` | Number of samples for the periodic WER check |
| `--adapter-dir` | `./adapters` | Local directory for adapter-only snapshots |

### DataLoader

| Flag | Default | Description |
|---|---|---|
| `--dataloader-num-workers` | `4` | Number of dataloader workers |
| `--dataloader-pin-memory / --no-dataloader-pin-memory` | `True` | Pin memory for GPU transfer |
| `--dataloader-persistent-workers / --no-dataloader-persistent-workers` | `True` | Keep workers alive between epochs |

### Augmentation

Probabilities for each augmentation stage. Set to `0` to disable.

| Flag | Default | Description |
|---|---|---|
| `--time-stretch-p` | `0.5` | Speed perturbation probability |
| `--time-stretch-min-rate` | `0.8` | Minimum speed perturbation factor |
| `--time-stretch-max-rate` | `1.25` | Maximum speed perturbation factor |
| `--noise-p` | `0.5` | Noise mixing probability |
| `--spec-augment-p` | `0.8` | SpecAugment probability |
| `--vtlp-p` | `0.5` | VTLP probability |


python3 train.py --batch-size 64 --grad-accum 1 --lora-dropout 0.05 --epochs 5 