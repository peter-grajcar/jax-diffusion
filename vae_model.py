import jax
import math
import jax.numpy as jnp
import flax.linen as nn


class Attention(nn.Module):
    num_heads: int

    @nn.compact
    def __call__(self, inputs, train=True):
        hidden = nn.BatchNorm(use_running_average=not train)(inputs)

        original_shape = hidden.shape
        hidden = jnp.reshape(hidden, (original_shape[0], original_shape[1] * original_shape[2], original_shape[3]))

        hidden = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, use_bias=False)(hidden, hidden, deterministic=not train)

        hidden = jnp.reshape(hidden, original_shape)

        hidden += inputs
        return hidden


class ResidualBlock(nn.Module):
    width: int

    @nn.compact
    def __call__(self, inputs, train=True):
        residual = nn.Conv(self.width, (1, 1))(inputs) if self.width != inputs.shape[-1] else inputs

        hidden = nn.BatchNorm(use_running_average=not train)(inputs)
        hidden = nn.swish(hidden)
        hidden = nn.Conv(self.width, (3, 3), padding="SAME", use_bias=False)(hidden)

        hidden += residual
        return hidden


class Downsample(nn.Module):
    factor: tuple[int, int]
    use_conv: bool = False

    @nn.compact
    def __call__(self, inputs):
        if self.use_conv:
            hidden = nn.Conv(inputs.shape[-1], (3, 3), strides=self.factor, padding="SAME")(inputs)
        else:
            hidden = nn.avg_pool(inputs, self.factor, strides=self.factor, padding="SAME")
        return hidden

class Upsample(nn.Module):
    factor: tuple[int, int]
    use_conv: bool = False

    @nn.compact
    def __call__(self, inputs):
        hidden = jax.image.resize(
            inputs,
            [inputs.shape[0], inputs.shape[1] * self.factor[0], inputs.shape[2] * self.factor[1], inputs.shape[3]],
             method="nearest"
        )
        if self.use_conv:
            hidden = nn.Conv(hidden.shape[-1], (3, 3), padding="SAME")(hidden)
        return hidden


class Encoder(nn.Module):
    z_dim: int
    channels: int
    stages: int
    stage_blocks: int
    attention_stages: int
    attention_heads: int

    @nn.compact
    def __call__(self, inputs, train=False):
        hidden = nn.Conv(self.channels, (3, 3), padding="SAME")(inputs)

        # Downsampling
        for i, factor in enumerate(self.stages):
            for _ in range(self.stage_blocks):
                hidden = ResidualBlock(self.channels << i)(hidden, train)

            if i >= len(self.stages) - self.attention_stages:
                hidden = Attention(num_heads=self.attention_heads)(hidden, train)

            hidden = Downsample(factor, use_conv=True)(hidden)

        # Middle
        for _ in range(self.stage_blocks):
            hidden = ResidualBlock(self.channels << len(self.stages))(hidden, train)

        # Outputs
        mean = nn.Conv(self.z_dim, (3, 3), padding="SAME", name=f"Encoder/Mean")(hidden)
        log_var = nn.Conv(self.z_dim, (3, 3), padding="SAME", name=f"Encoder/LogVariance")(hidden)

        return mean, log_var

class Decoder(nn.Module):
    z_dim: int
    channels: int
    out_channels: int
    stages: int
    stage_blocks: int
    attention_stages: int
    attention_heads: int

    @nn.compact
    def __call__(self, inputs, train=False):
        hidden = nn.Conv(self.channels, (1, 1), padding="SAME")(inputs)

        for i, factor in reversed(list(enumerate(self.stages))):
            hidden = Upsample(factor, use_conv=True)(hidden)

            if i >= len(self.stages) - self.attention_stages:
                hidden = Attention(num_heads=self.attention_heads)(hidden, train)

            for _ in range(self.stage_blocks):
                hidden = ResidualBlock(self.channels << i)(hidden, train)

        hidden = nn.BatchNorm(use_running_average=not train)(hidden)
        hidden = nn.swish(hidden)
        hidden = nn.Conv(self.out_channels, (3, 3), padding="SAME")(hidden)

        return hidden

class VAE(nn.Module):
    z_dim: int
    channels: int
    out_channels: int
    stages: int
    stage_blocks: int
    attention_stages: int
    attention_heads: int

    def setup(self):
        self.encoder = Encoder(
            self.z_dim,
            self.channels,
            self.stages,
            self.stage_blocks,
            self.attention_stages,
            self.attention_heads
        )
        self.decoder = Decoder(
            self.z_dim,
            self.channels,
            self.out_channels,
            self.stages,
            self.stage_blocks,
            self.attention_stages,
            self.attention_heads
        )

    def sample(self, mean, log_var, key):
        return mean + jnp.exp(log_var / 2) * jax.random.normal(key, mean.shape)

    def encode(self, inputs, train=False):
        return self.encoder(inputs, train=train)

    def decode(self, z, train=False):
        return self.decoder(z, train=train)

    def __call__(self, inputs, key, sample_posterior=True, train=False):
        mean, log_var = self.encode(inputs, train=train)
        if sample_posterior:
            z = self.sample(mean, log_var, key)
        else:
            z = mean
        return self.decode(z, train=train), mean, log_var


