import random
import numpy as np


class SpecAugment:
    """
    SpecAugment: frequency masking and time masking on mel spectrograms.

    Augmentation Parameters for policies
    -----------------------------------------
    Policy | F  | m_F |  T  |  p  | m_T
    -----------------------------------------
    None   |  0 |  -  |  0  |  -  |  -
    LB     | 27 |  1  | 100 | 1.0 | 1
    LD     | 27 |  2  | 100 | 1.0 | 2
    SM     | 15 |  2  |  70 | 0.2 | 2
    SS     | 27 |  2  |  70 | 0.2 | 2
    -----------------------------------------

    LB  : LibriSpeech basic
    LD  : LibriSpeech double
    SM  : Switchboard mild
    SS  : Switchboard strong
    F   : Frequency Mask parameter
    m_F : Number of Frequency masks
    T   : Time Mask parameter
    p   : Parameter for calculating upper bound for time mask
    m_T : Number of time masks
    """

    POLICIES = {
        "LB": {"F": 27, "m_F": 1, "T": 100, "p": 1.0, "m_T": 1},
        "LD": {"F": 27, "m_F": 2, "T": 100, "p": 1.0, "m_T": 2},
        "SM": {"F": 15, "m_F": 2, "T": 70, "p": 0.2, "m_T": 2},
        "SS": {"F": 27, "m_F": 2, "T": 70, "p": 0.2, "m_T": 2},
    }

    def __init__(self, policy: str = "LB"):
        if policy not in self.POLICIES:
            raise ValueError(f"Unknown policy '{policy}'. Choose from {list(self.POLICIES)}")
        params = self.POLICIES[policy]
        self.F: int = params["F"]
        self.m_F: int = params["m_F"]
        self.T: int = params["T"]
        self.p: float = params["p"]
        self.m_T: int = params["m_T"]

    def freq_mask(self, mel: np.ndarray) -> np.ndarray:
        """Apply m_F frequency masks to a (n_mels, time) spectrogram."""
        n_mels = mel.shape[0]
        for _ in range(self.m_F):
            f = int(np.random.uniform(0, self.F))
            f0 = random.randint(0, max(0, n_mels - f))
            mel[f0 : f0 + f, :] = 0
        return mel

    def time_mask(self, mel: np.ndarray) -> np.ndarray:
        """Apply m_T time masks to a (n_mels, time) spectrogram."""
        tau = mel.shape[1]
        for _ in range(self.m_T):
            t = int(np.random.uniform(0, min(self.T, int(self.p * tau))))
            if t == 0:
                continue
            t0 = random.randint(0, max(0, tau - t))
            mel[:, t0 : t0 + t] = 0
        return mel

    def __call__(self, mel: np.ndarray) -> np.ndarray:
        """
        Apply SpecAugment (freq + time masking) to a mel spectrogram.

        Args:
            mel: np.ndarray of shape (n_mels, time).

        Returns:
            Augmented mel spectrogram (same shape, modified in-place on a copy).
        """
        mel = mel.copy()
        mel = self.freq_mask(mel)
        mel = self.time_mask(mel)
        return mel
