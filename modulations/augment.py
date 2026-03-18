import random
from dataclasses import dataclass
from math import gcd

import librosa
import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import WhisperFeatureExtractor
from datasets import Audio as HFAudio
import io
from modulations.specaugment import SpecAugment
from modulations.vtlp import VTLP


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
@dataclass
class AugmentConfig:
    """All knobs for the augmentation pipeline."""

    sr: int = 16_000

    time_stretch_p: float = 0.5
    time_stretch_min_rate: float = 0.8
    time_stretch_max_rate: float = 1.25

    # ── waveform-level noise mixing ──
    noise_p: float = 0.5
    noise_snr_db_min: float = 5.0
    noise_snr_db_max: float = 30.0
    noise_peak_limit: float = 0.99

    # ── mel-level ──
    spec_augment_p: float = 0.8
    spec_policy: str = "LB"

    vtlp_p: float = 0.5
    vtlp_alpha_min: float = 0.8
    vtlp_alpha_max: float = 1.2

    # ── Whisper feature extractor ──
    whisper_model: str = "openai/whisper-medium.en"


# ──────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────
class Augment:
    """
    Data-augmentation pipeline for Whisper fine-tuning.

    Two stages:
        1. **Waveform-level** – speed perturbation + noise
           (applied independently per sample).
        2. **Mel-level**      – SpecAugment (freq/time masking) and VTLP
           (applied independently per spectrogram).

    Accepts a single 1-D waveform **or** a list of 1-D waveforms (batch).
    The Whisper feature extractor handles the list natively.
    """

    def __init__(self, config: AugmentConfig | None = None, noise_ds=None):
        self.cfg = config or AugmentConfig()

        # ── Whisper feature extractor (computes log-mel) ──
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self.cfg.whisper_model
        )

        # ── mel-level augmentations ──
        self.spec_aug = SpecAugment(policy=self.cfg.spec_policy)
        self.vtlp = VTLP()

        self.noise_ds = noise_ds

    @staticmethod
    def _factor_to_ratio(factor: float, precision: int = 1000) -> tuple[int, int]:
        """Convert a speed factor into reduced integer up/down ratio."""
        up = max(1, int(round(factor * precision)))
        down = precision
        div = gcd(up, down)
        return up // div, down // div

    def _speed_perturb(self, wf: np.ndarray) -> np.ndarray:
        """
        Speed perturb via polyphase resampling.
        factor > 1.0 => faster / shorter audio
        factor < 1.0 => slower / longer audio
        """
        if random.random() > self.cfg.time_stretch_p:
            return wf

        factor = random.uniform(
            self.cfg.time_stretch_min_rate, self.cfg.time_stretch_max_rate
        )
        up, down = self._factor_to_ratio(factor)
        return resample_poly(wf, up, down).astype(np.float32, copy=False)

    def _decode_audio(self, audio: HFAudio) -> np.ndarray:
        waveform, sr = librosa.load(
            io.BytesIO(audio['bytes']), dtype="float32"
        )
        return waveform, sr

    def _add_noise(self, wf: np.ndarray, sr: int) -> np.ndarray:
        """Mix a random noise clip into *wf* at a random SNR.

        The noise is randomly cropped (or looped) to match the signal length,
        then scaled so the mixture has the desired SNR.  A final hard-clip
        keeps the peak below ``noise_peak_limit``.
        """
        if self.noise_ds is None or random.random() > self.cfg.noise_p:
            return wf

        # ── pick & decode a random noise clip ──
        idx = random.randint(0, len(self.noise_ds["train"]) - 1)
        noise, noise_sr = self._decode_audio(self.noise_ds["train"][idx]["audio"])

        # ── resample noise to target sr if needed ──
        if noise_sr != sr:
            noise = librosa.resample(noise, orig_sr=noise_sr, target_sr=sr)

        # ── match lengths: loop if too short, random-crop if too long ──
        sig_len = len(wf)
        if len(noise) < sig_len:
            repeats = (sig_len // len(noise)) + 1
            noise = np.tile(noise, repeats)
            
        if len(noise) > sig_len:
            start = random.randint(0, len(noise) - sig_len)
            noise = noise[start : start + sig_len]

        # ── compute scale factor for target SNR ──
        target_snr_db = random.uniform(
            self.cfg.noise_snr_db_min, self.cfg.noise_snr_db_max
        )
        sig_power = np.mean(wf ** 2)
        noise_power = np.mean(noise ** 2)

        if noise_power == 0:
            return wf  # silent noise clip – nothing to add

        # desired_noise_power = sig_power / 10^(snr/10)
        scale = np.sqrt(sig_power / (noise_power * 10.0 ** (target_snr_db / 10.0)))
        mixed = wf + scale * noise

        # ── hard-clip to prevent clipping in downstream processing ──
        np.clip(mixed, -self.cfg.noise_peak_limit, self.cfg.noise_peak_limit, out=mixed)

        return mixed

    # ── waveform helpers ─────────────────────────────────────
    def augment_waveform(self, wf: np.ndarray, sr: int) -> np.ndarray:
        """Apply all waveform-level augmentations to **one** waveform."""
        wf = self._add_noise(wf, sr)
        aug = self._speed_perturb(wf)
        
        if len(aug)/sr > 30:
            return wf # just return unaugmented waveform if we exceed 30 seconds
        return aug

    def augment_waveform_batch(
        self, wfs: list[np.ndarray], sr: int
    ) -> list[np.ndarray]:
        """Apply waveform-level augmentations to a list of waveforms.

        Each waveform is augmented independently (with its own random
        parameters). Results may differ in length when speed perturbation
        is enabled.
        """
        augs = []
        for wf in wfs:
            augs.append(self.augment_waveform(wf, sr))
        return augs

    # ── mel helpers ───────────────────────────────────────────
    def compute_log_mel(self, wf: np.ndarray, sr: int) -> np.ndarray:
        """Compute a log-mel spectrogram for a single waveform."""
        features = self.feature_extractor(
            wf, sampling_rate=sr, return_tensors="np"
        )
        return features.input_features[0]  # (n_mels, T)

    def compute_log_mel_batch(
        self, wfs: list[np.ndarray], sr: int
    ) -> np.ndarray:
        """Compute log-mel spectrograms for a batch of waveforms.

        The Whisper feature extractor accepts a list of variable-length
        arrays and pads them to a common time dimension.

        Returns:
            np.ndarray of shape ``(B, n_mels, T)``.
        """
        features = self.feature_extractor(
            wfs, sampling_rate=sr, return_tensors="np"
        )
        return features.input_features  # (B, n_mels, T)

    def augment_mel(self, mel: np.ndarray) -> np.ndarray:
        """Apply mel-level augmentations to **one** spectrogram."""
        if random.random() < self.cfg.spec_augment_p:
            mel = self.spec_aug(mel)
        if random.random() < self.cfg.vtlp_p:
            alpha = random.uniform(
                self.cfg.vtlp_alpha_min, self.cfg.vtlp_alpha_max
            )
            mel = self.vtlp(mel, sampling_rate=self.cfg.sr, alpha=alpha)
            if isinstance(mel, torch.Tensor):
                mel = mel.numpy()
        return mel

    def augment_mel_batch(self, mels: np.ndarray) -> np.ndarray:
        """Apply mel-level augmentations independently to each spectrogram.

        Args:
            mels: Array of shape ``(B, n_mels, T)``.

        Returns:
            np.ndarray of the same shape.
        """
        augmented = [self.augment_mel(mel) for mel in mels]
        return np.stack(augmented)

    # ── full pipeline ─────────────────────────────────────────
    def __call__(
        self,
        wf: np.ndarray | list[np.ndarray],
        sr: int,
    ) -> (
        tuple[np.ndarray, np.ndarray]
        | tuple[list[np.ndarray], np.ndarray]
    ):
        """
        Run the full augmentation pipeline.

        Args:
            wf: **Single** 1-D waveform (float32, mono) **or** a
                ``list`` of 1-D waveforms for batched processing.
            sr: Sample rate (must be the same for all waveforms).

        Returns:
            *Single mode* –
                ``(augmented_waveform, augmented_log_mel)``
                where the mel has shape ``(n_mels, T)``.

            *Batch mode* –
                ``(list_of_augmented_waveforms, augmented_log_mels)``
                where the mel array has shape ``(B, n_mels, T)``.
        """
        single = isinstance(wf, np.ndarray) and wf.ndim == 1
        wfs: list[np.ndarray] = [wf] if single else list(wf)

        wfs_aug = self.augment_waveform_batch(wfs, sr)
        mels_aug = self.compute_log_mel_batch(wfs_aug, sr)
        mels_aug = self.augment_mel_batch(mels_aug)

        if single:
            return wfs_aug[0], mels_aug[0]
        return wfs_aug, mels_aug


# ──────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sounddevice as sd

    SR = 16_000

    # ── load two different clips to form a batch ──
    wf1, _ = librosa.load(librosa.ex("trumpet"), sr=SR)
    wf2, _ = librosa.load(librosa.ex("pistachio"), sr=SR)
    batch = [wf1, wf2]

    pipeline = Augment(AugmentConfig(sr=SR, num_workers=2))

    # ── single-sample call (backward compatible) ──
    wf_aug_single, mel_aug_single = pipeline(wf1, SR)
    print(
        f"[single] orig={len(wf1):,}  aug={len(wf_aug_single):,}  "
        f"mel={mel_aug_single.shape}"
    )

    # ── batched call ──
    wfs_aug, mels_aug = pipeline(batch, SR)
    print(f"\n[batch]  input size : {len(batch)}")
    for i, (wo, wa) in enumerate(zip(batch, wfs_aug)):
        print(
            f"  [{i}] orig={len(wo):,}  aug={len(wa):,}  "
            f"mel={mels_aug[i].shape}"
        )
    print(f"  mels_aug batch shape: {mels_aug.shape}")

    # ── play first augmented clip ──
    print("\nPlaying augmented clip 0 …")
    sd.play(wfs_aug[0], samplerate=SR)
    sd.wait()

    # ── plot all originals vs augmented ──
    n = len(batch)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        mel_orig = pipeline.compute_log_mel(batch[i], SR)

        img0 = axes[i, 0].imshow(
            mel_orig, aspect="auto", origin="lower", interpolation="none"
        )
        axes[i, 0].set_title(f"Original [{i}]")
        axes[i, 0].set_ylabel("Mel bin")
        fig.colorbar(img0, ax=axes[i, 0])

        img1 = axes[i, 1].imshow(
            mels_aug[i], aspect="auto", origin="lower", interpolation="none"
        )
        axes[i, 1].set_title(f"Augmented [{i}]")
        fig.colorbar(img1, ax=axes[i, 1])

    for ax in axes[-1]:
        ax.set_xlabel("Time frame")

    plt.tight_layout()
    plt.show()