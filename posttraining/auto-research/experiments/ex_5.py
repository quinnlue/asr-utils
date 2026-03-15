"""Experiment 5 – Transcription consistency normalization.

Fixes common missing-apostrophe contractions and other transcription
inconsistencies in **both** ref and pred so the Whisper normalizer can
properly expand them (e.g. "dont" → "don't" → "do not").
"""

import re

NAME = "ex_5_transcription_consistency"

# ── contraction restoration rules ────────────────────────────────────
# Map bare (no-apostrophe) forms to their canonical contractions.
# The Whisper normalizer then expands these contractions identically
# on both sides, eliminating false WER penalties.

_CONTRACTION_MAP: list[tuple[re.Pattern, str]] = [
    # Negations
    (re.compile(r"\bdont\b"),      "don't"),
    (re.compile(r"\bcant\b"),      "can't"),
    (re.compile(r"\bwont\b"),      "won't"),
    (re.compile(r"\bdidnt\b"),     "didn't"),
    (re.compile(r"\bwasnt\b"),     "wasn't"),
    (re.compile(r"\bisnt\b"),      "isn't"),
    (re.compile(r"\bwouldnt\b"),   "wouldn't"),
    (re.compile(r"\bshouldnt\b"), "shouldn't"),
    (re.compile(r"\bcouldnt\b"),   "couldn't"),
    (re.compile(r"\bhasnt\b"),     "hasn't"),
    (re.compile(r"\bhavent\b"),    "haven't"),
    (re.compile(r"\bhadnt\b"),     "hadn't"),
    (re.compile(r"\bwerent\b"),    "weren't"),
    (re.compile(r"\barent\b"),     "aren't"),
    (re.compile(r"\bdoesnt\b"),    "doesn't"),

    # Pronoun contractions
    (re.compile(r"\bweve\b"),      "we've"),
    (re.compile(r"\bive\b"),       "i've"),
    (re.compile(r"\btheyve\b"),    "they've"),
    (re.compile(r"\byouve\b"),     "you've"),

    (re.compile(r"\bwere\b(?=\s+(?:not|gonna|going|doing))"), "we're"),
    (re.compile(r"\btheyre\b"),    "they're"),
    (re.compile(r"\byoure\b"),     "you're"),

    (re.compile(r"\bhes\b"),       "he's"),
    (re.compile(r"\bshes\b"),      "she's"),
    (re.compile(r"\bwhats\b"),     "what's"),
    (re.compile(r"\bthats\b"),     "that's"),
    (re.compile(r"\btheres\b"),    "there's"),
    (re.compile(r"\bheres\b"),     "here's"),
    (re.compile(r"\bwhos\b"),      "who's"),
    (re.compile(r"\blets\b"),      "let's"),

    (re.compile(r"\bill\b"),       "i'll"),
    (re.compile(r"\btheyll\b"),    "they'll"),
    (re.compile(r"\byoull\b"),     "you'll"),
    (re.compile(r"\bwell\b(?=\s+(?:be|have|do|go|see|get|need|try))"), "we'll"),
    (re.compile(r"\bhell\b(?=\s+(?:be|have|do|go|see|get|need|try))"), "he'll"),
    (re.compile(r"\bshell\b(?=\s+(?:be|have|do|go|see|get|need|try))"), "she'll"),

    (re.compile(r"\bid\b(?=\s+(?:like|love|rather|say|want|prefer|be|have|go))"), "i'd"),
    (re.compile(r"\btheyd\b"),     "they'd"),
    (re.compile(r"\byoud\b"),      "you'd"),

    (re.compile(r"\bim\b"),        "i'm"),
]

# "its" at the start of a sentence / after punctuation → "it's"
_ITS_START_RE = re.compile(r"(?:^|(?<=[.!?,;]\s))its\b")
# "its" followed by a verb / adjective indicator → "it's"
_ITS_VERB_RE = re.compile(
    r"\bits\b(?=\s+(?:a|an|the|not|gonna|going|really|very|so|just|like|"
    r"been|being|called|made|got|getting|still|always|never|also|about|"
    r"pretty|kind|cool|fun|hard|good|bad|nice|big|small|hot|cold))"
)


def _normalize(text: str) -> str:
    for pattern, replacement in _CONTRACTION_MAP:
        text = pattern.sub(replacement, text)

    # Aggressive "its" → "it's" heuristics
    text = _ITS_START_RE.sub("it's", text)
    text = _ITS_VERB_RE.sub("it's", text)

    return text


# ── experiment entry point ───────────────────────────────────────────

def transform(df):
    """Restore missing apostrophes in both ref and pred."""
    df = df.copy()
    df["ref"] = df["ref"].apply(_normalize)
    df["pred"] = df["pred"].apply(_normalize)
    return df
