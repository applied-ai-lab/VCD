import flax.linen as nn


class Encoder(nn.Module):
    """ This is the module for CNN encoder.

    Attributes:
        latent_dim (int): the dimension of the latent space
    """

    latent_dim: int

    @nn.compact
    def __call__(self, x):
        """ Returns the mean and variance of the latent variable.

        Args:
            x (DeviceArray): the input image (assumed to be 128x128).

        Returns:
            mu (DeviceArray): The mean of the encoding.
            log_var (DeviceArray): The log variance of the encoding.
        """
        x = nn.Conv(
            features=16, kernel_size=(4, 4), strides=(2, 2), padding=((1, 1), (1, 1))
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=32, kernel_size=(4, 4), strides=(2, 2), padding=((0, 0), (0, 0))
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=64, kernel_size=(3, 3), strides=(2, 2), padding=((0, 0), (0, 0))
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=128, kernel_size=(4, 4), strides=(2, 2), padding=((0, 0), (0, 0))
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=256, kernel_size=(4, 4), strides=(2, 2), padding=((0, 0), (0, 0))
        )(x)
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        mu = nn.Dense(features=self.latent_dim)(x)
        log_var = nn.Dense(features=self.latent_dim)(x)
        return mu, log_var
