#!/usr/bin/env python3
import jax
import optax
import argparse
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ddim_train import blur_single, load_vae
from functools import partial
from itertools import chain, repeat
from ddim import TrainState, generate
from ddim_model import DiffusionModel
from dataset import get_mel_spectrum, normalise_images, denormalise_images
from flax.training import checkpoints


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", default="ckpt", type=str, help="Checkpoint directory.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--sampling_steps", default=50, type=int, help="Sampling steps.")
parser.add_argument("--input", type=str, help="Input file.")


def merge(batch):
    batch = np.transpose(batch, (1, 0, 2, 3))
    batch = np.reshape(batch, (batch.shape[0], -1, batch.shape[-1]))
    return batch

def split_frames(signal, frame_width, overlap):
    # signal shape: (n_mels, length, 1)

    # pad signal to be divisible by frame_width
    _, length, _ = signal.shape
    pad = length % frame_width
    signal = np.pad(signal, ((0, 0), (0, pad), (0, 0)), mode="edge")

    n_mels, length, _ = signal.shape
    n_frames = (length - frame_width) // (frame_width - overlap) + 1

    frames = np.zeros((n_frames, n_mels, frame_width, 1))
    for i in range(n_frames):
        frames[i] = signal[:, i*(frame_width-overlap):i*(frame_width-overlap)+frame_width]

    return frames

def overlap_add(frames, overlap, original_length):
    # frames shape: (n_frames, n_mels, frame_width, 1)
    frames = np.array(frames)

    n_frames, n_mels, frame_width, _ = frames.shape
    length = (n_frames - 1) * (frame_width - overlap) + frame_width

    signal = np.zeros((n_mels, length, 1))
    for i, frame in enumerate(frames):
        if i > 0:
            crossfade = np.linspace(0, 1, overlap)[:, None]
            frame[:, :overlap, :] *= crossfade
        if i < n_frames - 1:
            crossfade = np.linspace(1, 0, overlap)[:, None]
            frame[:, -overlap:, :] *= crossfade
        signal[:, i*(frame_width-overlap):i*(frame_width-overlap)+frame_width, :] += frame

    return signal[:, :original_length, :]

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    key = jax.random.PRNGKey(args.seed)

    ckpt = checkpoints.restore_checkpoint(args.ckpt_dir, None)

    # sigma=ckpt["blur"]["sigma"]
    # kernel_size=ckpt["blur"]["kernel_size"]
    # augment = [
    #    jax.vmap(partial(blur_single, sigma=5, kernel_size=5)),
    #    jax.vmap(partial(blur_single, sigma=5, kernel_size=7)),
    #    jax.vmap(partial(blur_single, sigma=5, kernel_size=11)),
    #]
    vae, vae_variables = load_vae("vae_ckpt-3")

    config = ckpt["config"]

    model = DiffusionModel(
        stages=config.stages,
        stage_blocks=config.stage_blocks,
        channels=config.channels,
        out_channels=1,
        attention_stages=config.attention_stages,
        attention_heads=config.attention_heads,
        scale_with_conv=config.scale_with_conv,
    )

    variables = ckpt["ema_variables"]
    mean, std = ckpt["normalisation_stats"]["mean"], ckpt["normalisation_stats"]["std"]
    frame_width = ckpt["frame_width"]
    overlap = frame_width // 4

    spectrum = get_mel_spectrum(args.input)[..., None]
    # original = np.array([spectrum[:, i*frame_width:(i+1)*frame_width, None] for i in range(spectrum.shape[1] // frame_width)])
    original_length = spectrum.shape[1]
    original = split_frames(spectrum, frame_width, overlap)

    mean, std = np.mean(spectrum), np.std(spectrum)
    print(mean, std)

    normalised = normalise_images(original, mean, std)
    key, noise_key, augment_key = jax.random.split(key, 3)
    noise = jax.random.normal(noise_key, normalised.shape)
    # augmented = jnp.concatenate([f(normalised) for f in augment], axis=-1)
    augmented, _, _ = vae.apply(vae_variables, normalised, augment_key, sample_posterior=False, train=False)

    # mean, std = -6.3, 1.9

    generated, diffusion_process = generate(model.apply, variables, noise, augmented, args.sampling_steps)
    diffusion_process = list(diffusion_process)
    diffusion_process.append(generated)

    print(optax.squared_error(generated, normalised).mean())
    denormalised = denormalise_images(np.squeeze(overlap_add(generated, overlap, original_length), -1), mean, std)
    np.save("zuzka.npy", denormalised, allow_pickle=False)

    n = len(diffusion_process)

    fig, ax = plt.subplots(4, 1, figsize=(7, 7), sharex=True, sharey=True)
    vmin, vmax = normalised.min(), normalised.max()
    ax[0].set_title("Original")
    ax[0].imshow(overlap_add(normalised, overlap, original_length), vmin=vmin, vmax=vmax, origin="lower")
    ax[1].set_title("Conditioning")
    ax[1].imshow(overlap_add(augmented[..., :1], overlap, original_length), vmin=vmin, vmax=vmax, origin="lower")
    ax[3].set_title("Generated")
    ax[3].imshow(overlap_add(generated, overlap, original_length), vmin=vmin, vmax=vmax, origin="lower")
    im = ax[2].imshow(overlap_add(diffusion_process[0], overlap, original_length), vmin=vmin, vmax=vmax, origin="lower")

    def animate(i):
        ax[2].set_title(f"t = {i / (n - 1):.2f}")
        im.set_data(overlap_add(diffusion_process[i], overlap, original_length))

    frames = chain(repeat(0, 10), range(n), repeat(n - 1, 10), reversed(range(n)))
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=100)
    plt.tight_layout()
    plt.show()
    # writer = animation.PillowWriter(fps=10)
    # ani.save("zuzka.webp", writer=writer)
    # ani.save("zuzka.mov")
