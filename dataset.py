#!/usr/bin/env python3
import sys
import jax
import jax.numpy as jnp
import librosa
import numpy as np
from torch.utils.data import IterableDataset


SAMPLING_RATE = 24000
HOP_LENGTH = 256
WIN_SIZE = 1024
N_FFT = 1024
N_MELS = 100
F_MIN = 0
F_MAX = 12000

MEL_BASIS = librosa.filters.mel(sr=SAMPLING_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX)
HANN_WINDOW = librosa.filters.get_window("hann", WIN_SIZE, fftbins=True)


@jax.jit
def normalise_images(images: jnp.ndarray, mean: float, std: float):
    return (images - mean) / std


@jax.jit
def denormalise_images(images: jnp.ndarray, mean: float, std: float):
    return images * std + mean


def dynamic_range_compression(x, c=1, clip_val=1e-5):
    return np.log(np.maximum(x, clip_val) * c)


def get_mel_spectrum(audio_path: str) -> np.ndarray:
    y, _ = librosa.load(audio_path, sr=SAMPLING_RATE)
    y = np.pad(y, (int((N_FFT - HOP_LENGTH) / 2), int((N_FFT - HOP_LENGTH) / 2)), mode="reflect")

    spec = librosa.stft(
        y,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_SIZE,
        window=HANN_WINDOW,
        center=False,
        pad_mode="reflect"
    )
    spec = np.abs(spec)

    spec = np.matmul(MEL_BASIS, spec)
    spec = dynamic_range_compression(spec)

    return spec


class MelDataset(IterableDataset):
    def __init__(self, path: str, frame_width: int=32):
        self.spectra = list(np.load(path).values())
        self.frame_width = frame_width
        self.n_mels = self.spectra[0].shape[0]

    def stats(self):
        all = np.concatenate(self.spectra, axis=1)
        return np.mean(all), np.std(all)

    def get_item(self):
        index = np.random.randint(0, len(self.spectra))
        spectrum = self.spectra[index]
        start = np.random.randint(0, spectrum.shape[1] - self.frame_width)
        end = start + self.frame_width
        return spectrum[:, start:end]

    def __iter__(self):
        def generator():
            while True:
                yield self.get_item()

        return generator()


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


if __name__ == "__main__":
    files = sys.argv[1:]

    spectra = []
    for file in files:
        print(file)
        spectrum = get_mel_spectrum(file)
        spectra.append(spectrum)

    np.savez_compressed("dataset", *spectra)

