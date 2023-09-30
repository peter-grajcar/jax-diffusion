#!/usr/bin/env python3
import jax
import optax
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from train import augment_single
from functools import partial
from ddim import TrainState, generate, normalise_images, denormalise_images
from model import DiffusionModel
from dataset import get_mel_spectrum
from flax.training import checkpoints


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", default="ckpt", type=str, help="Checkpoint directory.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--sampling_steps", default=50, type=int, help="Sampling steps.")
parser.add_argument("--input", type=str, help="Input file.")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    key = jax.random.PRNGKey(args.seed)

    ckpt = checkpoints.restore_checkpoint(args.ckpt_dir, None)

    sigma=ckpt["blur"]["sigma"]
    kernel_size=ckpt["blur"]["kernel_size"]
    augment = jax.vmap(partial(augment_single, sigma=sigma, kernel_size=kernel_size))

    model = DiffusionModel(
        stages=ckpt["stages"],
        stage_blocks=ckpt["stage_blocks"],
        channels=ckpt["channels"],
        out_channels=1,
    )

    variables = ckpt["ema_variables"]
    mean, std = ckpt["normalisation_stats"]["mean"], ckpt["normalisation_stats"]["std"]

    original = get_mel_spectrum(args.input)[:, :ckpt["frame_width"], None]
    normalised = normalise_images(original, mean, std)[None, ...]
    key, noise_key = jax.random.split(key, 2)
    noise = jax.random.normal(noise_key, normalised.shape)
    augmented = augment(normalised)

    generated, diffusion_process = generate(model.apply, variables, noise, augmented, args.sampling_steps)
    diffusion_process = list(diffusion_process)
    diffusion_process.append(generated)

    print(optax.squared_error(generated, normalised).mean())

    fig, ax = plt.subplots(1, 4)
    vmin, vmax = normalised.min(), normalised.max()
    ax[0].set_title("Original")
    ax[0].imshow(normalised[0], vmin=vmin, vmax=vmax, origin="lower")
    ax[1].set_title("Conditioning")
    ax[1].imshow(augmented[0], vmin=vmin, vmax=vmax, origin="lower")
    ax[3].set_title("Generated")
    ax[3].imshow(generated[0], vmin=vmin, vmax=vmax, origin="lower")
    im = ax[2].imshow(diffusion_process[0][0], vmin=vmin, vmax=vmax, origin="lower")

    def animate(i):
        ax[2].set_title(f"t = {i / len(diffusion_process):.2f}")
        im.set_data(diffusion_process[i][0])

    ani = animation.FuncAnimation(fig, animate, frames=len(diffusion_process), interval=100)
    plt.tight_layout()
    plt.show()
    ani.save("diffusion.gif")
