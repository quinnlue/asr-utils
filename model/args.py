import argparse

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Fine-tune Whisper with LoRA + augmentations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── data ──
    g = p.add_argument_group("Data")
    g.add_argument("--dataset", default="quinnlue/audio-cleaned-train",
                    help="HF dataset ID for train/val/test splits")
    g.add_argument("--noise-dataset", default="quinnlue/realclass",
                    help="HF dataset ID for noise samples")
    g.add_argument("--val-size", type=int, default=4800,
                    help="Number of validation samples to use (0 = all)")

    # ── model ──
    g = p.add_argument_group("Model")
    g.add_argument("--model", default="openai/whisper-medium.en",
                    help="Pretrained Whisper model ID")

    # ── LoRA ──
    g = p.add_argument_group("LoRA")
    g.add_argument("--lora-r", type=int, default=64,
                    help="LoRA rank")
    g.add_argument("--lora-alpha", type=int, default=128,
                    help="LoRA alpha")
    g.add_argument("--lora-dropout", type=float, default=0.0,
                    help="LoRA dropout")
    g.add_argument("--lora-target-modules", nargs="+",
                    default=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
                    help="LoRA target modules")

    # ── training ──
    g = p.add_argument_group("Training")
    g.add_argument("--per-device-train-batch-size", type=int, default=16)
    g.add_argument("--grad-accum", type=int, default=1,
                    help="Gradient accumulation steps")
    g.add_argument("--epochs", type=int, default=10)
    g.add_argument("--lr", type=float, default=3e-5,
                    help="Peak learning rate")
    g.add_argument("--lr-scheduler", default="cosine",
                    help="LR scheduler type")
    g.add_argument("--warmup-steps", type=int, default=2000)
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--adam-beta1", type=float, default=0.9)
    g.add_argument("--adam-beta2", type=float, default=0.98)
    g.add_argument("--optim", default="adamw_torch",
                    help="Optimizer name")
    g.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                    default=False)
    g.add_argument("--torch-compile", action=argparse.BooleanOptionalAction,
                    default=True, help="Enable torch.compile via Trainer")

    # ── eval / save ──
    g = p.add_argument_group("Evaluation & checkpointing")
    g.add_argument("--save-steps", type=int, default=1000)
    g.add_argument("--save-total-limit", type=int, default=64,
                    help="Max checkpoints to keep on disk")
    g.add_argument("--logging-steps", type=int, default=25)
    g.add_argument("--generation-max-length", type=int, default=128)

    # ── I/O ──
    g = p.add_argument_group("Output")
    g.add_argument("--output-dir", default="./whisper-medium-finetune",
                    help="Local checkpoint directory")
    g.add_argument("--report-to", default="wandb",
                    help="Experiment tracker (wandb, tensorboard, none)")

    # ── resume ──
    g = p.add_argument_group("Resumption")
    g.add_argument("--resume-from", default=None,
                    help="Local checkpoint path to resume from "
                         "(e.g. './whisper-medium-finetune/checkpoint-1500'). "
                         "Omit to train from scratch.")

    # ── dataloader ──
    g = p.add_argument_group("DataLoader")
    g.add_argument("--dataloader-num-workers", type=int, default=8)
    g.add_argument("--dataloader-pin-memory", action=argparse.BooleanOptionalAction,
                    default=True)
    g.add_argument("--dataloader-persistent-workers", action=argparse.BooleanOptionalAction,
                    default=True)

    # ── augmentation overrides ──
    g = p.add_argument_group("Augmentation")
    g.add_argument("--pitch-shift-p", type=float, default=0.25)
    g.add_argument("--pitch-min-semitones", type=float, default=-4.0)
    g.add_argument("--pitch-max-semitones", type=float, default=2.0)
    g.add_argument("--time-stretch-p", type=float, default=0.25)
    g.add_argument("--time-stretch-min-rate", type=float, default=0.8)
    g.add_argument("--time-stretch-max-rate", type=float, default=1.25)
    g.add_argument("--time-stretch-leave-length-unchanged",
                    action=argparse.BooleanOptionalAction, default=False)
    g.add_argument("--noise-p", type=float, default=0.5)
    g.add_argument("--noise-snr-db-min", type=float, default=5.0)
    g.add_argument("--noise-snr-db-max", type=float, default=30.0)
    g.add_argument("--noise-peak-limit", type=float, default=0.99)
    g.add_argument("--spec-augment-p", type=float, default=0.8)
    g.add_argument("--spec-policy", default="LB",
                    help="SpecAugment policy")
    g.add_argument("--vtlp-p", type=float, default=0.5)
    g.add_argument("--vtlp-alpha-min", type=float, default=0.8)
    g.add_argument("--vtlp-alpha-max", type=float, default=1.2)

    return p.parse_args(argv)
