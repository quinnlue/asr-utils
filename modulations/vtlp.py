import numpy as np
import torch


class VTLP:
    """
    Vocal Tract Length Perturbation (VTLP) applied directly to
    log-mel spectrograms (e.g. Whisper preprocessor output).

    Warps the frequency (mel) bins of the spectrogram using a
    piecewise-linear function controlled by alpha.
    """

    @staticmethod
    def get_scale_factors(
        n_mels: int,
        sampling_rate: int,
        fhi: float = 4800.0,
        alpha: float = 0.9,
    ) -> np.ndarray:
        """
        Compute warped frequency mapping for mel bins.

        Args:
            n_mels:         Number of mel bins.
            sampling_rate:  Audio sampling rate in Hz.
            fhi:            Upper frequency boundary for warping (Hz).
            alpha:          Warp factor. <1 shorter tract, >1 longer.

        Returns:
            np.ndarray of warped bin indices in [0, n_mels-1].
        """
        freqs = np.linspace(0, 1, n_mels) * sampling_rate / 2.0
        half_sr = sampling_rate / 2.0
        scale = fhi * min(alpha, 1.0)
        f_boundary = scale / alpha

        factors = np.where(
            freqs <= f_boundary,
            freqs * alpha,
            half_sr
            - (half_sr - scale) / (half_sr - scale / alpha) * (half_sr - freqs),
        )

        # Normalize to mel-bin index range [0, n_mels - 1]
        factors *= (n_mels - 1) / factors.max()
        return factors

    @staticmethod
    def warp_mel(mel_spec: np.ndarray, factors: np.ndarray) -> np.ndarray:
        """
        Warp a mel spectrogram along the frequency axis.

        Args:
            mel_spec:  2-D array of shape (n_mels, time).
            factors:   Warped bin indices of shape (n_mels,).

        Returns:
            Warped mel spectrogram, same shape and dtype.
        """
        n_mels, _ = mel_spec.shape
        warped = np.zeros_like(mel_spec)

        for i in range(n_mels):
            if i == 0 or i + 1 >= n_mels:
                warped[i, :] += mel_spec[i, :]
            else:
                pos = int(np.floor(factors[i]))
                frac = factors[i] - pos
                warped[pos, :] += (1.0 - frac) * mel_spec[i, :]
                if pos + 1 < n_mels:
                    warped[pos + 1, :] += frac * mel_spec[i, :]

        return warped

    def __call__(
        self,
        input_features: "np.ndarray | torch.Tensor",
        sampling_rate: int = 16000,
        alpha: float = 0.9,
        fhi: float = 4800.0,
    ) -> "np.ndarray | torch.Tensor":
        """
        Apply VTLP to a mel spectrogram.

        Args:
            input_features:  Array/Tensor of shape (batch, n_mels, time)
                             or (n_mels, time).
            sampling_rate:   Sampling rate (default 16000).
            alpha:           Warp factor (e.g. 0.8–1.2). 1.0 = identity.
            fhi:             Upper freq boundary for piecewise warp (Hz).

        Returns:
            Warped features, same type and shape as input.
        """
        is_tensor = isinstance(input_features, torch.Tensor)
        if is_tensor:
            device = input_features.device
            arr = input_features.cpu().numpy()
        else:
            arr = np.asarray(input_features, dtype=np.float32)

        # Handle batched and unbatched
        single = arr.ndim == 2
        if single:
            arr = arr[np.newaxis, ...]  # (1, n_mels, time)

        batch_size, n_mels, _ = arr.shape
        factors = self.get_scale_factors(n_mels, sampling_rate, fhi=fhi, alpha=alpha)

        warped_batch = np.empty_like(arr)
        for b in range(batch_size):
            warped_batch[b] = self.warp_mel(arr[b], factors)

        if single:
            warped_batch = warped_batch.squeeze(0)

        if is_tensor:
            return torch.from_numpy(warped_batch).to(device)
        return warped_batch
