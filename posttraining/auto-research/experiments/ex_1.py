"""Experiment 1 – Remove decoder loops from predictions."""

import re

NAME = "ex_1_remove_decoder_loops"


# ── helpers ──────────────────────────────────────────────────────────

def _loop_pattern(min_repeats: int, max_phrase_words: int) -> str:
    """Build a regex that matches any 1–N word phrase repeated consecutively."""
    return (
        r"(\b[\w']+(?:\s+[\w']+){0," + str(max_phrase_words - 1) + r"}?)"
        r"(\s+\1){" + str(min_repeats - 1) + r",}\b"
    )


def detect_decoder_loop(
    text: str,
    min_repeats: int = 4,
    max_phrase_words: int = 10,
) -> bool:
    """
    Return True if `text` contains any word or phrase repeated
    `min_repeats`+ times consecutively.
    """
    pattern = _loop_pattern(min_repeats, max_phrase_words)
    return bool(re.search(pattern, text))


def remove_decoder_loops(
    text: str,
    min_repeats: int = 3,
    max_phrase_words: int = 10,
) -> str:
    """Replace every loop with a single occurrence of the phrase."""
    pattern = _loop_pattern(min_repeats, max_phrase_words)
    prev = None
    while prev != text:
        prev = text
        text = re.sub(pattern, r"\1", text)
    return re.sub(r" +", " ", text).strip()


# ── experiment entry point ───────────────────────────────────────────

def transform(df):
    """Remove decoder loops from the predicted text."""
    df = df.copy()
    df["pred"] = df["pred"].apply(remove_decoder_loops)
    return df
