import argparse

from datasets import load_dataset
from huggingface_hub import login


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


if __name__ == "__main__":
    main()
