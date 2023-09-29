#!/usr/bin/env python3
import jax
import optax
import argparse
import jax.numpy as jnp
from model import DiffusionModel
from pprint import pprint
from flax.training import train_state
from typing import Any, Dict
from tqdm.auto import tqdm
from functools import partial

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--channels", default=32, type=int, help="CNN channels in the first stage.")
parser.add_argument("--dataset", default="oxford_flowers102", type=str, help="Image64 dataset to use.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--downscale", default=8, type=int, help="Conditional downscale factor.")
parser.add_argument("--ema", default=0.999, type=float, help="Exponential moving average momentum.")
parser.add_argument("--epoch_batches", default=1_000, type=int, help="Batches per epoch.")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
parser.add_argument("--loss", default="MeanAbsoluteError", type=str, help="Loss object to use.")
parser.add_argument("--plot_each", default=None, type=int, help="Plot generated images every such epoch.")
parser.add_argument("--sampling_steps", default=50, type=int, help="Sampling steps.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--stage_blocks", default=2, type=int, help="ResNet blocks per stage.")
parser.add_argument("--stages", default=4, type=int, help="Stages to use.")

args = parser.parse_args([] if "__file__" not in globals() else None)

model = DiffusionModel(
    stages=args.stages,
    stage_blocks=args.stage_blocks,
    channels=args.channels,
)

key = jax.random.PRNGKey(args.seed)
key, init_key = jax.random.split(key)

variables = model.init(init_key, jnp.ones([1, 64, 64, 3]), jnp.ones([1, 64, 64, 3]), jnp.ones([1, 1, 1, 1]), train=False)
params = variables["params"]
batch_stats = variables["batch_stats"]

class TrainState(train_state.TrainState):
    batch_stats: Any
    key: jax.random.PRNGKey
    ema_variables: Dict

key, train_key = jax.random.split(key)
state = TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optax.adamw(1e-3, b1=0.9, b2=0.999, eps=1e-8),
    batch_stats=batch_stats,
    key=train_key,
    ema_variables=variables.copy().unfreeze(),
    ema_momentum=args.ema,
)


@jax.jit
def diffusion_rates(times: jnp.ndarray):
    starting_angle, final_angle = 0.025, jnp.pi / 2 - 0.025
    signal_rates = jnp.cos((1 - times) * starting_angle + times * final_angle)[..., None, None, None]
    noise_rates = jnp.sin((1 - times) * starting_angle + times * final_angle)[..., None, None, None]
    return signal_rates, noise_rates


@jax.jit
def normalise_images(images: jnp.ndarray, mean: float, std: float):
    return (images - mean) / std


@jax.jit
def denormalise_images(images: jnp.ndarray, mean: float, std: float):
    images = images * std + mean
    images = jnp.clip(images, 0, 255)
    return images.astype(jnp.uint8)


@jax.jit
def train_step(state: TrainState, batch: tuple[jnp.ndarray, jnp.ndarray]):
    images, conditioning = batch
    images = normalise_images(images, ..., ...)

    key, noise_key, time_key = jax.random.split(state.key, 3)
    state = state.replace(key=key)
    noises = jax.random.normal(noise_key, images.shape[0])
    times = jax.random.uniform(time_key, images.shape[:1])

    signal_rates, noise_rates = diffusion_rates(times)
    noisy_images = signal_rates * images + noise_rates * noises

    def loss_fn(params):
        predicted_noises, updates = state.apply_fn(
            {"params": state.params, "batch_stats": state.batch_stats},
            noisy_images,
            conditioning,
            noise_rates,
            train=True,
        )
        loss = optax.squared_error(noises, predicted_noises).mean()
        return loss, (predicted_noises, updates)

    # Gradient update
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (predicted_noises, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates["batch_stats"])

    # Exponential moving average update
    optax.incremental_update(
        {"params": state.params, "batch_stats": state.batch_stats},
        state.ema_variables,
        state.ema_momentum
    )
    state = state.replace(ema_variables=ema_variables)

    return state, loss

