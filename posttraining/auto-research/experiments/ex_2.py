"""Experiment 2 – Children's informal speech normalization.

Normalizes common informal / colloquial variants that the Whisper
EnglishTextNormalizer does not handle.  Applied to **both** ref and pred
so it acts as an extended normalizer (these are genuinely equivalent
forms in children's speech).
"""

import re

NAME = "ex_2_informal_speech_norm"

# ── replacement table ────────────────────────────────────────────────
# Each entry is (compiled_regex, replacement_string).
# Order matters: longer / more specific patterns first.

_RULES: list[tuple[re.Pattern, str]] = [
    # Multi-word contractions
    (re.compile(r"\bdunno\b"),   "do not know"),
    (re.compile(r"\bgonna\b"),   "going to"),      # Whisper handles, but belt-and-suspenders
    (re.compile(r"\bwanna\b"),   "want to"),
    (re.compile(r"\bgotta\b"),   "got to"),
    (re.compile(r"\bhafta\b"),   "have to"),
    (re.compile(r"\blemme\b"),   "let me"),
    (re.compile(r"\bgimme\b"),   "give me"),
    (re.compile(r"\bkinda\b"),   "kind of"),
    (re.compile(r"\bsorta\b"),   "sort of"),

    # Causal "'cause" / sentence-initial "cause" → because
    (re.compile(r"\b'cause\b"),  "because"),
    (re.compile(r"^cause\b"),    "because"),        # sentence-initial bare "cause"

    # Pronoun contractions
    (re.compile(r"\b'em\b"),     "them"),

    # Affirmative / negative variants
    (re.compile(r"\byeah\b"),    "yes"),
    (re.compile(r"\byep\b"),     "yes"),
    (re.compile(r"\byup\b"),     "yes"),
    (re.compile(r"\bnah\b"),     "no"),
    (re.compile(r"\bnope\b"),    "no"),
]


def _normalize(text: str) -> str:
    """Apply all informal-speech normalization rules to *text*."""
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    # Collapse any double spaces introduced by replacements
    return re.sub(r" {2,}", " ", text).strip()


# ── experiment entry point ───────────────────────────────────────────

def transform(df):
    """Normalize informal speech in both ref and pred."""
    df = df.copy()
    df["ref"] = df["ref"].apply(_normalize)
    df["pred"] = df["pred"].apply(_normalize)
    return df
