import flax.linen as nn
import jax.numpy as jnp


class Decoder(nn.Module):
    """ This is the module for deconvolution decoder."""

    @nn.compact
    def __call__(self, x: jnp.DeviceArray) -> jnp.DeviceArray:
        """ Returns the decoded image.

        Args:
            x (DeviceArray): the input encoding.

        Returns:
            The decoded image (3x128x128).
        """
        x = nn.relu(nn.Dense(features=1024)(x))
        x = x.reshape((x.shape[0], 1, 1, 1024))
        x = nn.relu(
            nn.ConvTranspose(features=128, kernel_size=(5, 5), strides=(2, 2))(x)
        )
        x = nn.relu(
            nn.ConvTranspose(features=64, kernel_size=(5, 5), strides=(4, 4))(x)
        )
        x = nn.relu(
            nn.ConvTranspose(features=32, kernel_size=(6, 6), strides=(4, 4))(x)
        )
        x = nn.relu(
            nn.ConvTranspose(features=16, kernel_size=(3, 3), strides=(4, 4))(x)
        )
        x = nn.ConvTranspose(features=3, kernel_size=(2, 2), strides=(1, 1))(x)
        return nn.sigmoid(x)
