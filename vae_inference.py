#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from vae_model import VAE
from dataset import MelDataset, numpy_collate, normalise_images, denormalise_images
from torch.utils.data import DataLoader
from flax.training import checkpoints


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    np.random.seed(42)

    ckpt = checkpoints.restore_checkpoint("vae_ckpt", target=None)

    config = ckpt["config"]

    vae = VAE(
        config.z_dim,
        confi.channels,
        1,
        stages,
        config.stage_blocks,
        config.attention_stages,
        config.attention_heads,
        config.scale_with_conv,
    )

    variables = ckpt["ema_variables"]
    mean, std = ckpt["normalisation_stats"]["mean"], ckpt["normalisation_stats"]["std"]

    dataset = MelDataset("data/dataset.npz")
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=numpy_collate,
    )
    x = next(iter(data_loader))

    x = x[..., None]
    x = normalise_images(x, mean, std)

    y, _, _ = vae.apply(variables, x, key, sample_posterior=True, train=False)
    y = denormalise_images(y, mean, std)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x[0, ..., 0], origin="lower")
    ax[1].imshow(y[0, ..., 0], origin="lower")
    plt.show()

