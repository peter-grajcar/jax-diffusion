#!/usr/bin/env python3
import jax
import optax
import argparse
import numpy as np
import jax.numpy as jnp
from vae import TrainState, train_step
from dataset import MelDataset, numpy_collate, normalise_images
from tqdm.auto import tqdm
from itertools import islice
from vae_model import VAE
from flax.training import checkpoints
from torch.utils.data import DataLoader
from config import load_config

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epoch_batches", default=1_000, type=int, help="Batches per epoch.")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--ema_momentum", default=0.9, type=float, help="Exponential moving average momentum.")
parser.add_argument("--ckpt_dir", default="vae_ckpt", type=str, help="Checkpoint directory.")
parser.add_argument("--config", default="vae_config.json", type=str, help="Checkpoint directory.")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    key = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)

    config = load_config(args.config)

    vae = VAE(
        config.z_dim,
        config.channels,
        1,
        config.stages,
        config.stage_blocks,
        config.attention_stages,
        config.attention_heads,
    )

    key, init_key, dummy_key = jax.random.split(key, 3)
    table = vae.tabulate(init_key, jnp.ones([1, 100, 32, 1]), dummy_key, sample_posterior=True, train=True, depth=2)
    print(table)

    # Init model params
    key, init_key, dummy_key = jax.random.split(key, 3)
    variables = vae.init(init_key, jnp.ones([1, 100, 32, 1]), dummy_key, sample_posterior=True, train=True)

    # Create optimiser
    schedule = optax.exponential_decay(1e-3, args.epoch_batches, 0.98)
    optimiser = optax.adamw(schedule, weight_decay=0.004, b1=0.9, b2=0.999, eps=1e-7)

    # Create training state
    key, train_key = jax.random.split(key)
    state = TrainState.create(
        apply_fn=vae.apply,
        params=variables["params"],
        tx=optimiser,
        key=train_key,
        z_dim=config.z_dim,
        batch_stats=variables["batch_stats"],
        ema_variables=variables.copy(),
        ema_momentum=args.ema_momentum,
    )

    # Load dataset
    dataset = MelDataset("data/dataset.npz")
    data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=numpy_collate,
    )
    mean, std = dataset.stats()

    # Train
    min_loss = jnp.inf
    for epoch in range(args.epochs):
        bar = tqdm(
            islice(data_loader, args.epoch_batches),
            total=args.epoch_batches,
            desc=f"Epoch {epoch + 1}/{args.epochs}"
        )

        mean_loss = 0
        for i, images in enumerate(bar):
            images = images[..., None]
            images = normalise_images(images, mean, std)

            state, loss, reconstruction_loss, latent_loss = train_step(state, images)
            mean_loss += loss
            bar.set_postfix(loss=f"{mean_loss / (i + 1):.4f}, recon={reconstruction_loss:.4f}, kl={latent_loss:.4f}")

        mean_loss /= args.epoch_batches
        if mean_loss < min_loss:
            min_loss = mean_loss
            checkpoints.save_checkpoint(
                args.ckpt_dir,
                target={
                    "params": state.params,
                    "batch_stats": state.batch_stats,
                    "ema_variables": state.ema_variables,
                    "config": config,
                    "normalisation_stats": {"mean": mean, "std": std},
                },
                step=state.step
            )


