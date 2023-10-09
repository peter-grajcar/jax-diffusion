#!/usr/bin/env python3
import jax
import optax
import jax.numpy as jnp
from typing import Callable
from flax.training import train_state

class TrainState(train_state.TrainState):
    batch_stats: dict
    key: jax.random.PRNGKey
    z_dim: int
    ema_variables: dict
    ema_momentum: float

@jax.jit
def kl_divergence(mean, log_var):
    return 0.5 * (1 + log_var - mean ** 2 - jnp.exp(log_var)).sum(axis=-1)

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        key, apply_key = jax.random.split(state.key)
        (decoded, mean, log_var), updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            batch,
            apply_key,
            sample_posterior=True,
            mutable=["batch_stats"]
        )

        reconstruction_loss = jnp.abs(decoded - batch).mean()
        latent_loss = -kl_divergence(mean, log_var).mean()
        loss = reconstruction_loss * batch.shape[1] * batch.shape[2] + latent_loss * state.z_dim

        return loss, (key, reconstruction_loss, latent_loss, updates)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(state.params)
    key, reconstruction_loss, latent_loss, updates = aux

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates["batch_stats"])
    state = state.replace(key=key)

    # Exponential moving average update
    ema_variables = optax.incremental_update(
        {"params": state.params, "batch_stats": state.batch_stats},
        state.ema_variables,
        state.ema_momentum
    )
    state = state.replace(ema_variables=ema_variables)

    return state, loss, reconstruction_loss, latent_loss

