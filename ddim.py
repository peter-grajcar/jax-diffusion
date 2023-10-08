import jax
import optax
import jax.numpy as jnp
from typing import Any, Callable
from tqdm.auto import tqdm
from functools import partial
from flax.training import train_state

class TrainState(train_state.TrainState):
    batch_stats: Any
    key: jax.random.PRNGKey
    ema_variables: dict
    ema_momentum: float


@jax.jit
def diffusion_rates(times: jnp.ndarray):
    starting_angle, final_angle = 0.025, jnp.pi / 2 - 0.025
    signal_rates = jnp.cos((1 - times) * starting_angle + times * final_angle)[..., None, None, None]
    noise_rates = jnp.sin((1 - times) * starting_angle + times * final_angle)[..., None, None, None]
    return signal_rates, noise_rates

@jax.jit
def train_step(state: TrainState, batch: tuple[jnp.ndarray, jnp.array, jnp.ndarray], mask: jnp.ndarray=None):
    images, noises, conditioning = batch

    key, time_key = jax.random.split(state.key, 2)
    state = state.replace(key=key)
    times = jax.random.uniform(time_key, images.shape[:1])

    signal_rates, noise_rates = diffusion_rates(times)
    noisy_images = signal_rates * images + noise_rates * noises

    def loss_fn(params):
        predicted_noises, updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            noisy_images,
            conditioning,
            noise_rates,
            train=True,
            mutable=["batch_stats"],
        )
        loss = jnp.abs(noises - predicted_noises)
        if mask is not None:
            loss = loss * mask
        loss = loss.mean()
        return loss, (predicted_noises, updates)

    # Gradient update
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (predicted_noises, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates["batch_stats"])

    # Exponential moving average update
    ema_variables = optax.incremental_update(
        {"params": state.params, "batch_stats": state.batch_stats},
        state.ema_variables,
        state.ema_momentum
    )
    state = state.replace(ema_variables=ema_variables)

    return state, loss


# @partial(jax.jit, static_argnums=[0, 4])
def generate(apply_fn: Callable, variables: dict, initial_noise: jnp.ndarray, conditioning: jnp.ndarray, num_steps: int):
    images = initial_noise
    diffusion_process = []

    steps = jnp.linspace(jnp.ones(initial_noise.shape[:1]), jnp.zeros(initial_noise.shape[:1]), num_steps + 1)

    for times, next_times in zip(steps[:-1], steps[1:]):
        diffusion_process.append(images)

        signal_rates, noise_rates = diffusion_rates(times)

        predicted_noises = apply_fn(
            variables,
            images,
            conditioning,
            noise_rates,
            train=False,
        )

        denoised_images = (images - noise_rates * predicted_noises) / signal_rates

        next_signal_rates, next_noise_rates = diffusion_rates(next_times)

        images = next_signal_rates * denoised_images + next_noise_rates * predicted_noises

    return images, diffusion_process

