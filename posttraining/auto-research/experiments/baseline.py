"""Baseline – no transformation, scores the raw predictions."""

NAME = "baseline"


def transform(df):
    """Identity transform: return data as-is."""
    return df
