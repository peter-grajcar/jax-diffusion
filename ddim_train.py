#!/usr/bin/env python3
import argparse
import dm_pix
import optax
import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from ddim import TrainState, train_step, generate
from ddim_model import DiffusionModel
from dataset import MelDataset, numpy_collate, normalise_images, denormalise_images
from functools import partial
from itertools import islice
from torch.utils.data import DataLoader, random_split
from flax.training import checkpoints
from tqdm.auto import tqdm
from pprint import pprint


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--attention_stages", default=0, type=int, help="Number of stages with self attention.")
parser.add_argument("--attention_heads", default=8, type=int, help="number of self attention heads.")
parser.add_argument("--channels", default=32, type=int, help="CNN channels in the first stage.")
parser.add_argument("--dataset", default="oxford_flowers102", type=str, help="Image64 dataset to use.")
parser.add_argument("--downscale", default=8, type=int, help="Conditional downscale factor.")
parser.add_argument("--ema", default=0.999, type=float, help="Exponential moving average momentum.")
parser.add_argument("--epoch_batches", default=1_000, type=int, help="Batches per epoch.")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
parser.add_argument("--loss", default="MeanAbsoluteError", type=str, help="Loss object to use.")
parser.add_argument("--plot_each", default=None, type=int, help="Plot generated images every such epoch.")
parser.add_argument("--sampling_steps", default=50, type=int, help="Sampling steps.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--stage_blocks", default=2, type=int, help="ResNet blocks per stage.")
parser.add_argument("--stages", default=2, type=int, help="Stages to use.")
parser.add_argument("--blur_sigma", default=3, type=float, help="Gaussian blur sigma.")
parser.add_argument("--blur_kernel", default=5, type=int, help="Gaussian blur kernel size.")
parser.add_argument("--mask_width", default=3, type=int, help="Mask width.")
parser.add_argument("--ckpt_dir", default="ckpt", type=str, help="Checkpoint directory.")

def blur_single(img: jnp.array, sigma: float, kernel_size: int):
    img = jnp.pad(img, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2), (0, 0)), mode="edge")
    return dm_pix.gaussian_blur(img, sigma, kernel_size, padding="VALID")

def _mask_single(img: jnp.ndarray, noise: jnp.ndarray, mask_width: int):
    # img: [height, width, channels]
    width = img.shape[1]
    mask_pos = width // 2
    return jnp.concatenate([
        jnp.where(jnp.abs(jnp.arange(width)[None, :, None] - mask_pos) < mask_width, noise, img),
        jnp.where(jnp.abs(jnp.arange(width)[None, :, None] - mask_pos) < mask_width, jnp.ones_like(img), jnp.zeros_like(img)),
    ], axis=-1)

def mask_single(img: jnp.ndarray, key: jax.random.PRNGKey, mask_width: int):
    # img: [height, width, channels]
    width = img.shape[1]
    mask_pos = width // 2
    key_a, key_b = jax.random.split(key)
    uniform = jax.random.uniform(key_a, img.shape)
    normal = jax.random.normal(key_b, img.shape)
    middle = jnp.abs(jnp.arange(width)[None, :, None] - mask_pos) < mask_width
    return jnp.where(jnp.logical_and(middle, uniform > 0.2), normal, img)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # augment = jax.vmap(partial(mask_single, mask_width=args.mask_width))
    augment = [
        jax.vmap(partial(blur_single, sigma=5, kernel_size=5)),
        jax.vmap(partial(blur_single, sigma=5, kernel_size=7)),
        jax.vmap(partial(blur_single, sigma=5, kernel_size=11)),
    ]

    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    # Initialise model
    model = DiffusionModel(
        stages=args.stages,
        stage_blocks=args.stage_blocks,
        channels=args.channels,
        out_channels=1,
        attention_stages=args.attention_stages,
        attention_heads=args.attention_heads,
    )
    key, init_key = jax.random.split(key)
    variables = model.init(init_key, jnp.ones([1, 64, 64, 1]), jnp.ones([1, 64, 64, len(augment)]), jnp.ones([1, 1, 1, 1]), train=False)
    params = variables["params"]
    batch_stats = variables["batch_stats"]

    # pprint(jax.tree_util.tree_map(jnp.shape, params))

    # Create optimiser
    schedule = optax.cosine_decay_schedule(1e-3, args.epochs * args.epoch_batches, 1e-5)
    optimiser = optax.adamw(schedule, weight_decay=0.004, b1=0.9, b2=0.999, eps=1e-7)

    # Create training state
    key, train_key = jax.random.split(key)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimiser,
        batch_stats=batch_stats,
        key=train_key,
        ema_variables=variables.copy(),
        ema_momentum=args.ema,
    )

    # Load dataset
    dataset = MelDataset("data/dataset.npz")
    data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=numpy_collate,
    )
    mean, std = dataset.stats()

    # Prepare development set
    dev = next(iter(data_loader))
    dev = dev[..., None]
    dev = normalise_images(dev, mean, std)
    key, noise_key, augment_key = jax.random.split(key, 3)
    dev_noise = jax.random.normal(noise_key, dev.shape)
    augmented = jnp.concatenate([f(dev) for f in augment], axis=-1)
    dev = (dev_noise, dev, augmented)

    generate = jax.jit(generate, static_argnums=[0, 4])

    best_mse = np.inf
    for epoch in range(args.epochs):
        bar = tqdm(
            islice(data_loader, args.epoch_batches),
            total=args.epoch_batches,
            desc=f"Epoch {epoch + 1}/{args.epochs}"
        )

        # Training
        for images in bar:
            images = images[..., None]
            images = normalise_images(images, mean, std)
            key, augment_key, noise_key = jax.random.split(key, 3)
            noises = jax.random.normal(noise_key, images.shape)
            augmented = jnp.concatenate([f(images) for f in augment], axis=-1)

            # print(state.params["ResidualBlock_0"]["Conv_1"]["kernel"][0])
            batch = (images, noises, augmented)
            state, loss = train_step(state, batch)
            bar.set_postfix(loss=f"{loss:.4f}")

        # Validation
        noise, original, augmented = dev
        generated, _ = generate(state.apply_fn, state.ema_variables, noise, augmented, args.sampling_steps)
        val_mse = optax.squared_error(original, generated).mean()
        print(f"Validation MSE: {val_mse:.4f}")

        if val_mse < best_mse:
            best_mse = val_mse
            checkpoints.save_checkpoint(
                args.ckpt_dir,
                target={
                    "stages": args.stages,
                    "stage_blocks": args.stage_blocks,
                    "attention_stages": args.attention_stages,
                    "attention_heads": args.attention_heads,
                    "channels": args.channels,
                    "params": state.params,
                    "batch_stats": state.batch_stats,
                    "ema_variables": state.ema_variables,
                    "normalisation_stats": {"mean": mean, "std": std},
                    "frame_width": dataset.frame_width,
                    # "blur": {"sigma": args.blur_sigma, "kernel_size": args.blur_kernel},
                    # "mask_width": args.mask_width,
                },
                step=state.step,
                keep=3
            )

