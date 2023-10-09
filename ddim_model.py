import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import jax

class SinusoidalEmbedding(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, inputs):
        i = jnp.arange(self.dim // 2)
        sin_embeddings = jnp.sin(2 * np.pi * inputs / 20 ** (2 * i / self.dim))
        cos_embeddings = jnp.cos(2 * np.pi * inputs / 20 ** (2 * i / self.dim))
        embeddings = jnp.concatenate([sin_embeddings, cos_embeddings], axis=-1)
        return embeddings


class Attention(nn.Module):
    num_heads: int

    @nn.compact
    def __call__(self, inputs, conditioning, train=True):
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
    def __call__(self, inputs, noise_embeddings, train=True):
        residual = nn.Conv(self.width, (1, 1))(inputs) if self.width != inputs.shape[-1] else inputs

        hidden = nn.BatchNorm(use_running_average=not train)(inputs)
        hidden = nn.swish(hidden)
        hidden = nn.Conv(self.width, (3, 3), padding="SAME", use_bias=False)(hidden)

        noise_embeddings = nn.Dense(self.width)(noise_embeddings)
        noise_embeddings = nn.swish(noise_embeddings)
        hidden += noise_embeddings

        hidden = nn.BatchNorm(use_running_average=not train)(hidden)
        hidden = nn.swish(hidden)
        hidden = nn.Conv(self.width, (3, 3), padding="SAME", use_bias=False, kernel_init=nn.initializers.zeros)(hidden)

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



class DiffusionModel(nn.Module):
    stages: int
    stage_blocks: int
    channels: int
    out_channels: int = 3
    attention_stages: int = 0
    attention_heads: int = 8
    scale_with_conv: bool = False

    @nn.compact
    def __call__(self, inputs, conditioning, noise_rates, train=True):
        noise_embeddings = SinusoidalEmbedding(self.channels)(noise_rates)

        inputs = jnp.concatenate([inputs, conditioning], axis=-1)
        hidden = nn.Conv(self.channels, (3, 3), padding="SAME")(inputs)

        outputs = []
        for i, factor in enumerate(self.stages):
            for _ in range(self.stage_blocks):
                hidden = ResidualBlock(self.channels << i)(hidden, noise_embeddings, train)
                outputs.append(hidden)

            if i >= len(self.stages) - self.attention_stages:
                hidden = Attention(num_heads=self.attention_heads)(hidden, conditioning, train)

            hidden = Downsample(factor, use_conv=self.scale_with_conv)(hidden)

        for _ in range(self.stage_blocks):
            hidden = ResidualBlock(self.channels << len(self.stages))(hidden, noise_embeddings, train)

        if self.attention_stages > 0:
            hidden = Attention(num_heads=self.attention_heads)(hidden, conditioning, train)

        for i, factor in reversed(list(enumerate(self.stages))):
            hidden = Upsample(factor, use_conv=self.scale_with_conv)(hidden)

            if i >= len(self.stages) - self.attention_stages:
                hidden = Attention(num_heads=self.attention_heads)(hidden, conditioning, train)

            for _ in range(self.stage_blocks):
                hidden = jnp.concatenate([hidden, outputs.pop()], axis=-1)
                hidden = ResidualBlock(self.channels << i)(hidden, noise_embeddings, train)

        assert len(outputs) == 0

        outputs = nn.BatchNorm(use_running_average=not train)(hidden)
        outputs = nn.swish(outputs)
        outputs = nn.Conv(self.out_channels, (3, 3), padding="SAME", kernel_init=nn.initializers.zeros)(outputs)

        return outputs

