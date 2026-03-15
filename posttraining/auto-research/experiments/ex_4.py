"""Experiment 4 – Aggressive decoder-loop removal with tiered thresholds.

Builds on ex_1's approach with an additional aggressive tier for very
long predictions that are almost certainly decoder hallucinations:

  1. All phrases (1-10 words): min_repeats=3  (same as ex_1)
  2. If prediction exceeds 80 words (beyond any reference in the
     dataset), lower all thresholds to min_repeats=2 – the model is
     hallucinating and any repeated phrase should be collapsed.
"""

import re

NAME = "ex_4_aggressive_loop_removal"


# ── core loop-removal ────────────────────────────────────────────────

def _loop_pattern(min_repeats: int, max_phrase_words: int) -> str:
    """Regex matching a 1–*max_phrase_words* word phrase repeated
    *min_repeats*+ times consecutively."""
    return (
        r"(\b[\w']+(?:\s+[\w']+){0," + str(max_phrase_words - 1) + r"}?)"
        r"(\s+\1){" + str(min_repeats - 1) + r",}\b"
    )


def _remove_loops(text: str, min_repeats: int, max_phrase_words: int) -> str:
    pattern = _loop_pattern(min_repeats, max_phrase_words)
    prev = None
    while prev != text:
        prev = text
        text = re.sub(pattern, r"\1", text)
    return text


def _clean(text: str) -> str:
    # Tier 1: standard loop removal (same as ex_1)
    text = _remove_loops(text, min_repeats=3, max_phrase_words=10)

    # Tier 2: if still very long, aggressively collapse any repeated
    # phrase (min 2 repeats) – the model is hallucinating
    if len(text.split()) > 80:
        text = _remove_loops(text, min_repeats=2, max_phrase_words=10)

    return re.sub(r" +", " ", text).strip()


# ── experiment entry point ───────────────────────────────────────────

def transform(df):
    """Aggressively remove decoder loops from predictions."""
    df = df.copy()
    df["pred"] = df["pred"].apply(_clean)
    return df
