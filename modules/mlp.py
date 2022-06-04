import flax.linen as nn
import jax.numpy as jnp


class MLP(nn.Module):
    """ This is a module for general conditional probabilities parameterised by MLP.

    The conditional distributions are modelled as independent Gaussian distributions with learnt mean and variance.

    Attributes:
        out_dim (int): The dimension of the target.
        hidden_dim (int): The dimension of the hidden layers.
    """

    out_dim: int = 9
    hidden_dim: int = 64

    @nn.compact
    def __call__(
        self, x: jnp.DeviceArray, mask: None
    ) -> tuple[jnp.DeviceArray, jnp.DeviceArray]:
        """ Returns the mean and log variance of the conditional distribution.

        Args:
            x (DeviceArray): The variable to be conditioned on.
            mask (DeviceArray): Not used in this fully-connected implementation.

        Returns:
            mu (DeviceArray): The predicted mean of the conditional distribution.
            log_var (DevicdArray): The predicted log variance fo the conditional state distribution.
        """
        log_var = self.param("log_var", nn.initializers.zeros, (self.out_dim,))
        x = nn.relu(nn.Dense(features=self.hidden_dim)(x))
        x = nn.relu(nn.Dense(features=self.hidden_dim)(x))
        mu = nn.Dense(features=self.out_dim)(x)
        return mu, jnp.broadcast_to(log_var, (x.shape[0], log_var.shape[0]))
