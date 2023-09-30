#!/usr/bin/env python3
import sys
import librosa
import matplotlib.pyplot as plt
from dataset import get_mel_spectrum
from train import augment_single

spectrum = get_mel_spectrum(sys.argv[1])
spectrum = spectrum[..., None]
augmented = augment_single(spectrum, 3, 5)

print(spectrum.shape)
print(augmented.shape)

fig, ax = plt.subplots(2, 1)
ax[0].imshow(spectrum)
ax[1].imshow(augmented)
plt.tight_layout()
plt.show()

