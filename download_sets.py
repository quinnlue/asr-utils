from datasets import load_dataset
from huggingface_hub import login

login()

ds = load_dataset("quinnlue/audio-cleaned")
noise_ds = load_dataset("quinnlue/realclass")