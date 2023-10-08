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

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epoch_batches", default=1_000, type=int, help="Batches per epoch.")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--stages", default=2, type=int, help="ResNet blocks per stage.")
parser.add_argument("--stage_blocks", default=2, type=int, help="ResNet blocks per stage.")
parser.add_argument("--channels", default=8, type=int, help="Number of channels.")
parser.add_argument("--attention_stages", default=0, type=int, help="Number of stages with self attention.")
parser.add_argument("--attention_heads", default=8, type=int, help="number of self attention heads.")
parser.add_argument("--z_dim", default=128, type=int, help="Stages to use.")
parser.add_argument("--ckpt_dir", default="vae_ckpt", type=str, help="Checkpoint directory.")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    key = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)

    vae = VAE(
        args.z_dim,
        args.channels,
        1,
        args.stages,
        args.stage_blocks,
        args.attention_stages,
        args.attention_heads,
    )

    # Init model params
    key, init_key, dummy_key = jax.random.split(key, 3)
    variables = vae.init(init_key, jnp.ones([1, 100, 32, 1]), dummy_key, sample_posterior=True, train=True)

    # Create optimiser
    schedule = optax.cosine_decay_schedule(1e-3, args.epochs * args.epoch_batches, 1e-5)
    optimiser = optax.adamw(schedule, weight_decay=0.004, b1=0.9, b2=0.999, eps=1e-7)

    # Create training state
    key, train_key = jax.random.split(key)
    state = TrainState.create(
        apply_fn=vae.apply,
        params=variables["params"],
        tx=optimiser,
        key=train_key,
        z_dim=args.z_dim,
        batch_stats=variables["batch_stats"],
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
                    "z_dim": state.z_dim,
                    "stages": args.stages,
                    "stage_blocks": args.stage_blocks,
                    "attention_stages": args.attention_stages,
                    "attention_heads": args.attention_heads,
                    "channels": args.channels,
                    "out_channels": 1,
                    "normalisation_stats": {"mean": mean, "std": std},
                },
                step=state.step
            )


