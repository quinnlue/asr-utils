"""Experiment 3 – Non-speech artifact cleanup.

Removes IPA characters, garbled / repeated-character tokens, and other
non-ASCII garbage from predictions.  Only modifies ``pred``.
"""

import re
import unicodedata

NAME = "ex_3_artifact_cleanup"

# ── helpers ──────────────────────────────────────────────────────────

# Characters we allow in cleaned predictions (ASCII letters, digits,
# basic punctuation, whitespace).
_ALLOWED_RE = re.compile(r"[^a-zA-Z0-9\s'\-.,!?]")

# Detect "words" that are a short substring repeated 3+ times
# e.g. "balkbalkbalk" → repeating unit "balk" × 3
_GARBLED_RE = re.compile(r"\b(.{1,6}?)\1{2,}\b")


def _clean_text(text: str) -> str:
    """Strip non-ASCII chars and garbled tokens from *text*."""
    # Normalize unicode to NFKD, then drop non-ASCII
    text = unicodedata.normalize("NFKD", text)
    text = _ALLOWED_RE.sub("", text)

    # Remove garbled repeated-character words
    text = _GARBLED_RE.sub("", text)

    # Collapse whitespace
    return re.sub(r" {2,}", " ", text).strip()


# ── experiment entry point ───────────────────────────────────────────

def transform(df):
    """Clean non-speech artifacts from predictions."""
    df = df.copy()
    df["pred"] = df["pred"].apply(_clean_text)
    return df
