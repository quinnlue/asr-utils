"""Experiment Combo – All post-processing transforms stacked.

Applies every transform in sequence:
  1. Non-speech artifact cleanup  (ex_3)
  2. Decoder loop removal – aggressive  (ex_4, superset of ex_1)
  3. Informal speech normalization  (ex_2)
  4. Transcription consistency normalization  (ex_5)

Order rationale:
  - Clean garbage first so loop detection works on real words.
  - Remove loops before linguistic normalization so normalizers
    operate on shorter, cleaner text.
"""

from experiments.ex_2 import transform as _t2
from experiments.ex_3 import transform as _t3
from experiments.ex_4 import transform as _t4
from experiments.ex_5 import transform as _t5

NAME = "ex_combo_all_transforms"


def transform(df):
    """Apply all transforms in sequence."""
    df = _t3(df)   # artifact cleanup
    df = _t4(df)   # aggressive loop removal
    df = _t2(df)   # informal speech normalization
    df = _t5(df)   # transcription consistency
    return df

