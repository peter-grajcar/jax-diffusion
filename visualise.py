#!/usr/bin/env python3
import sys
import librosa
import jax
import matplotlib.pyplot as plt
from dataset import get_mel_spectrum
from train import blur_single, mask_single

key = jax.random.PRNGKey(42)

spectrum = get_mel_spectrum(sys.argv[1])
spectrum = spectrum[..., None]
spectrum = jax.numpy.array(spectrum)

fig, ax = plt.subplots(4, 1, sharex=True, sharey=True)
ax[0].imshow(spectrum, origin='lower')
ax[1].imshow(blur_single(spectrum, 10, 5), origin='lower')
ax[2].imshow(blur_single(spectrum, 7, 7), origin='lower')
ax[3].imshow(blur_single(spectrum, 3, 11), origin='lower')
plt.show()

