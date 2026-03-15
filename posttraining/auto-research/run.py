"""
Auto-research runner.

Discovers every experiment in the experiments/ folder, applies its
transform to the data, scores WER, and prints a comparison table.

To add a new experiment, just create a .py file in experiments/ with:
    NAME = "my_experiment"
    def transform(df) -> df: ...
"""

import pandas as pd

from experiments import discover
from metric import score_wer

DATA_PATH = "output.csv"


def main():
    df = pd.read_csv(DATA_PATH)
    experiments = discover()

    print(f"{'Experiment':<40} {'WER':>10}")
    print("-" * 52)

    last_transformed = None
    for exp in experiments:
        transformed = exp.transform(df.copy())
        wer = score_wer(transformed["ref"], transformed["pred"])
        print(f"{exp.name:<40} {wer:>10.6f}")
        last_transformed = transformed

    if last_transformed is not None:
        last_transformed.to_csv("output_combo.csv", index=False)
        print(f"\nSaved combo output to output_combo.csv")


if __name__ == "__main__":
    main()
