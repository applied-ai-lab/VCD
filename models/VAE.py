from models.sequence_model import BaseSequenceModel, KL_div
from modules.decoders import Decoder
from modules.encoders import Encoder
import flax
import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax


class VAE(nn.Module):
    latent_dim: int = 16

    @nn.compact
    def __call__(self, img, rng):
        mu, logvar = Encoder(self.latent_dim)(img)
        z = BaseSequenceModel.reparameterize(rng, mu, logvar)
        recon = Decoder()(z)
        return mu, logvar, recon

    def init_train_state(self, rng, batch, lr):
        model = self
        params = model.init(rng, batch[0], rng)
        tx = optax.adam(learning_rate=lr)
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    def load_train_state(self, rng, batch, pretrained_encoder, pretrained_decoder):
        model = self
        params = flax.core.unfreeze(model.init(rng, batch[0], rng))
        params["params"]["Decoder_0"] = pretrained_decoder
        params["params"]["Encoder_0"] = pretrained_encoder
        params = flax.core.freeze(params)
        tx = optax.adam(learning_rate=0.0001)
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=tx)

    @staticmethod
    @jax.jit
    def train(state, data, rng):
        # training data is a tuple (images [batch, H,W,C], action, rewards)
        def loss_fn(params):
            outputs = state.apply_fn(params, data[0], rng)
            recon = outputs[2]
            recon_mse = ((recon - data[0]) ** 2).mean(axis=0).sum()
            kl = (
                KL_div(
                    outputs[0],
                    outputs[1],
                    jnp.zeros_like(outputs[0]),
                    jnp.zeros_like(outputs[1]),
                )
                .mean(axis=0)
                .sum()
            )
            return (recon_mse + kl), (recon_mse, kl)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return loss[1], state

    @staticmethod
    @jax.jit
    def evaluate(state, data, rng):
        def loss_fn(params):
            outputs = state.apply_fn(params, data[0], rng)
            recon = outputs[2]
            recon_mse = ((recon - data[0]) ** 2).mean(axis=0).sum()
            kl = (
                KL_div(
                    outputs[0],
                    outputs[1],
                    jnp.zeros_like(outputs[0]),
                    jnp.zeros_like(outputs[1]),
                )
                .mean(axis=0)
                .sum()
            )
            return (recon_mse + kl), (recon_mse, kl)

        return loss_fn(state.params)[1]
