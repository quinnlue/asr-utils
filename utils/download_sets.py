import argparse

from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoModelForSpeechSeq2Seq

def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace datasets")
    parser.add_argument("datasets", nargs="+", help="HuggingFace dataset names to download")
    parser.add_argument("--token", required=True, help="HuggingFace API token")
    args = parser.parse_args()

    login(token=args.token)

    for name in args.datasets:
        print(f"Downloading {name}...")
        load_dataset(name)
        print(f"Done: {name}")

def download_set(name):
    ds = load_dataset(name)
    ds.save_to_disk(f"datasets/{name}")


def download_model(name, adapter_name=None):
    model = AutoModelForSpeechSeq2Seq.from_pretrained(name, device_map="cpu")
    if adapter_name:
        model.load_adapter(adapter_name)
        model = model.merge_and_unload()

    model.save_pretrained(f"models/{name}")


if __name__ == "__main__":
    main()
