"""
Audio augmentation pipeline for ASR training.

Provides configurable waveform-level and spectrogram-level augmentations:
  - Speed perturbation   (tempo change, preserves pitch)
  - Pitch perturbation   (pitch shift in semitones)
  - Volume perturbation  (gain change in dB)
  - VTLN                 (vocal tract length perturbation via frequency warping)
  - SpecAugment          (time & frequency masking on mel spectrograms)

Each augmentation is gated by an independent probability (0.0 = never, 1.0 = always).
Parameters are sampled uniformly from configurable ranges on every call.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Tuple

import librosa
import numpy as np
import torch
import torchaudio.transforms as T


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AugmentConfig:
    """All augmentation knobs in one place.

    Set any ``*_prob`` to 0.0 to disable that augmentation entirely.
    """

    # -- Waveform-level augmentations --

    # Speed perturbation (time-stretch without pitch change)
    speed_prob: float = 0.3
    speed_range: Tuple[float, float] = (0.9, 1.1)

    # Pitch perturbation (semitones)
    pitch_prob: float = 0.3
    pitch_range_semitones: Tuple[float, float] = (-2.0, 2.0)

    # Volume / gain perturbation (dB)
    volume_prob: float = 0.3
    volume_range_db: Tuple[float, float] = (-6.0, 6.0)

    # Vocal Tract Length Perturbation (VTLN) warp factor
    vtln_prob: float = 0.3
    vtln_warp_range: Tuple[float, float] = (0.9, 1.1)

    # -- Spectrogram-level augmentation (SpecAugment) --

    spec_augment_prob: float = 0.5
    n_time_masks: int = 2
    time_mask_param: int = 40       # max frames masked per time-mask
    n_freq_masks: int = 2
    freq_mask_param: int = 27       # max mel-bins masked per freq-mask


# ---------------------------------------------------------------------------
# Waveform-level helpers
# ---------------------------------------------------------------------------

def _speed_perturb(waveform: np.ndarray, sr: int, config: AugmentConfig) -> np.ndarray:
    """Time-stretch the waveform (changes speed/duration, preserves pitch)."""
    rate = random.uniform(*config.speed_range)
    if abs(rate - 1.0) < 1e-3:
        return waveform
    return librosa.effects.time_stretch(waveform, rate=rate)


def _pitch_perturb(waveform: np.ndarray, sr: int, config: AugmentConfig) -> np.ndarray:
    """Shift pitch by a random number of semitones."""
    n_steps = random.uniform(*config.pitch_range_semitones)
    if abs(n_steps) < 0.05:
        return waveform
    return librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)


def _volume_perturb(waveform: np.ndarray, _sr: int, config: AugmentConfig) -> np.ndarray:
    """Apply a random gain change in dB."""
    db = random.uniform(*config.volume_range_db)
    gain = 10.0 ** (db / 20.0)
    return (waveform * gain).astype(waveform.dtype)


def _vtln_warp(waveform: np.ndarray, sr: int, config: AugmentConfig) -> np.ndarray:
    """Vocal Tract Length Perturbation via piecewise-linear frequency warping.

    Operates in the STFT domain:
      1. Compute magnitude + phase via STFT.
      2. Build a piecewise-linear warp map on the frequency axis.
      3. Resample magnitude bins according to the warp map.
      4. Reconstruct waveform via iSTFT.
    """
    from scipy.interpolate import interp1d

    alpha = random.uniform(*config.vtln_warp_range)
    if abs(alpha - 1.0) < 1e-3:
        return waveform

    n_fft = 512
    hop_length = 128

    stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)

    n_bins = mag.shape[0]
    orig_freqs = np.linspace(0, 1, n_bins)

    # Piecewise-linear warp: f_warped = f * alpha  (clamped at Nyquist)
    warped_freqs = np.clip(orig_freqs * alpha, 0, 1)

    # Interpolate each time frame
    warped_mag = np.zeros_like(mag)
    for t in range(mag.shape[1]):
        interp_fn = interp1d(
            warped_freqs, mag[:, t],
            kind="linear", bounds_error=False, fill_value=0.0,
        )
        warped_mag[:, t] = interp_fn(orig_freqs)

    warped_stft = warped_mag * np.exp(1j * phase)
    result = librosa.istft(warped_stft, hop_length=hop_length, length=len(waveform))
    return result.astype(waveform.dtype)


# ---------------------------------------------------------------------------
# Public waveform augmentation entry-point
# ---------------------------------------------------------------------------

def augment_waveform(
    waveform: np.ndarray,
    sr: int,
    config: AugmentConfig,
) -> np.ndarray:
    """Apply waveform-level augmentations in sequence, each coin-flipped independently.

    Args:
        waveform: 1-D float32 numpy array (mono, 16 kHz expected).
        sr: Sample rate.
        config: Augmentation configuration.

    Returns:
        Augmented waveform (same dtype, may differ in length if speed-perturbed).
    """
    if random.random() < config.speed_prob:
        waveform = _speed_perturb(waveform, sr, config)

    if random.random() < config.pitch_prob:
        waveform = _pitch_perturb(waveform, sr, config)

    if random.random() < config.volume_prob:
        waveform = _volume_perturb(waveform, sr, config)

    if random.random() < config.vtln_prob:
        waveform = _vtln_warp(waveform, sr, config)

    return waveform


# ---------------------------------------------------------------------------
# Spectrogram-level augmentation (SpecAugment)
# ---------------------------------------------------------------------------

def augment_spectrogram(
    spec_tensor: torch.Tensor,
    config: AugmentConfig,
) -> torch.Tensor:
    """Apply SpecAugment (time + frequency masking) to a batch of mel spectrograms.

    Args:
        spec_tensor: Tensor of shape (B, n_mels, time) — e.g. (B, 80, 3000).
        config: Augmentation configuration.

    Returns:
        Masked spectrogram tensor of the same shape.
    """
    if random.random() >= config.spec_augment_prob:
        return spec_tensor

    spec_aug = T.SpecAugment(
        n_time_masks=config.n_time_masks,
        time_mask_param=config.time_mask_param,
        n_freq_masks=config.n_freq_masks,
        freq_mask_param=config.freq_mask_param,
        iid_masks=True,
        zero_masking=True,
    )
    return spec_aug(spec_tensor)
