# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "jiwer>=4.0.0",
#     "pandas>=2.3.3",
#     "transformers>=5.0.0",
# ]
# ///
import re
import string
from pathlib import Path
from unicodedata import normalize
from maps import *

import jiwer
import pandas as pd
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

##################### Normalization Functions #####################



_SPACE_RE = re.compile(r"\s+")


def normalize_ipa(s: str) -> str:
    """
    Normalize IPA for CER:
      - NFC normalize
      - Delete tie bars and stress marks (translate)
      - Decompose nasalized vowels to base+tilde (translate)
      - Normalize rhotics to ɚ (translate)
      - Remove ASCII punctuation
      - Map affricate digraphs to ligatures
      - Collapse whitespace
    """
    # Canonical composition
    s = normalize("NFC", s)

    # Translations
    s = s.translate(_TRANSLATION)

    # Map digraphs to affricate ligatures
    # Note: affricate ligatures are retired from IPA, but preferred by us for CER
    s = s.replace("tʃ", "ʧ").replace("dʒ", "ʤ")

    # Flatten whitespace
    s = _SPACE_RE.sub(" ", s).strip()
    return s


###################### Scoring Functions #####################


def validate_ipa_characters(s: str, raise_error=True) -> bool:
    """
    Validate that all characters in the IPA string are in the accepted set
    after normalization.

    Args:
        s (str): The IPA string to validate.
        raise_error (bool): If True, raise ValueError on invalid characters.
                            If False, just return False.
    Returns:
        bool: True if all characters are valid, False otherwise.
    """
    s = normalize_ipa(s)
    invalid_chars = set([c for c in s if c not in VALID_IPA_CHARS])

    if invalid_chars and raise_error:
        raise ValueError(
            "Invalid IPA characters found: " + ", ".join([f"'{c}'" for c in invalid_chars])
        )

    return not invalid_chars


def score_ipa_cer(actual, predicted) -> float:
    """
    Calculate the IPA Character Error Rate (IPA-CER) between predicted and
    reference sequences.

    Each predicted and reference string is first normalized with
    ``normalize_ipa``, which performs:
        - Unicode NFC normalization
        - Deletion of tie bars, stress marks, and ASCII punctuation
        - Decomposition of nasalized vowels into base vowel + tilde
        - Normalization of rhotic vowels to ɚ
        - Mapping of affricate digraphs (e.g. tʃ, dʒ) to ligatures
        - Whitespace collapsing

    CER is calculated as the total number of substitutions (S), deletions (D),
    and insertions (I) needed to transform the predicted text into the reference
    text, divided by the total number of characters (N) in the reference text.
    Lower is better.

    Args:
        actual: Sequence of actual (reference) IPA transcription strings.
        predicted: Sequence of predicted IPA transcription strings.

    Returns:
        float: The corpus-level CER score.
    """
    normalized_preds = [normalize_ipa(text) for text in predicted]
    normalized_refs = [normalize_ipa(text) for text in actual]
    results = jiwer.cer(normalized_refs, normalized_preds)
    return results


def score_wer(actual, predicted):
    """
    Calculate the Word Error Rate (WER) between predicted and actual sequences.

    Word Error Rate (WER) on normalized English transcripts.
    Predicted and reference strings are first passed through Whisper's
    English Text Normalizer before computing error rate at the word level.

    WER is calculated as the total number of substitutions (S), deletions (D),
    and insertions (I) needed to transform the predicted text into the reference
    text, divided by the total number of words (N) in the reference text.
    Lower is better.

    Args:
        actual: Sequence of actual (reference) transcription strings.
        predicted: Sequence of predicted transcription strings.

    Returns:
        float: The WER score.
    """
    normalizer = EnglishTextNormalizer(english_spelling_normalizer)
    normalized_preds = [normalizer(text) for text in predicted]
    normalized_refs = [normalizer(text) for text in actual]
    results = jiwer.wer(normalized_refs, normalized_preds)
    return results


def score_jsonl(path_to_predicted: Path, path_to_actual: Path, metric="wer") -> float:
    """
    Calculate WER and IPA-CER between predicted and actual transcriptions.
    Assumes predictions and labels are stored in competition submission format,
    i.e. jsonl files with an "utterance_id" and "orthographic_text" or "phonetic_text" field.

    Args:
        path_to_predicted (Path): Path to the jsonl file containing predicted transcriptions.
        path_to_actual (Path): Path to the jsonl file containing actual transcriptions.
        metric (str): The metric to compute. Either "wer" or "ipa_cer".

    Returns:
        float: The computed score.
    """
    if metric == "wer":
        metric_func = score_wer
    elif metric == "ipa_cer":
        metric_func = score_ipa_cer
    else:
        raise ValueError(f"Metric must be one of 'wer' or 'ipa_cer', got {metric}")

    predicted = pd.read_json(path_to_predicted, lines=True).set_index("utterance_id").sort_index()
    actual = pd.read_json(path_to_actual, lines=True).set_index("utterance_id").sort_index()

    text_field = "orthographic_text" if metric == "wer" else "phonetic_text"
    return metric_func(actual[text_field], predicted[text_field])


####################### English Spelling Normalizer ################

# Required by the Whisper English normalizer (used in the WER metric).
# Vendored from:
# https://github.com/huggingface/open_asr_leaderboard/blob/79d4760168013fafcfd4814bb521538bf008e828/normalizer/english_abbreviations.py


if __name__ == "__main__":
    # if used as test script take predicted and actual from command line args
    import sys

    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <predicted_jsonl> <actual_jsonl>")
        sys.exit(1)

    path_to_predicted = Path(sys.argv[1])
    path_to_actual = Path(sys.argv[2])

    # then, determine based on first line if ortho (wer) or phonetic (ipa_cer)
    try:
        with path_to_actual.open("r", encoding="utf-8") as f:
            first_line = f.readline()

        if '"orthographic_text"' in first_line:
            metric = "wer"
        elif '"phonetic_text"' in first_line:
            metric = "ipa_cer"
        else:
            print(
                "Error: Could not determine metric. JSONL must contain 'orthographic_text' or 'phonetic_text'."
            )
            sys.exit(1)
    except Exception as e:
        print(f"Error reading {path_to_actual}: {e}")
        sys.exit(1)

    # then print what is being scored and run scoring
    print(f"Scoring {metric.upper()} between {path_to_predicted} and {path_to_actual}...")
    result = score_jsonl(path_to_predicted, path_to_actual, metric=metric)
    print(f"{metric.upper()}: {result}")
