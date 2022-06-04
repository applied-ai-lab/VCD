import flax.linen as nn
import jax.numpy as jnp


class TransitionRNN(nn.Module):
    r""" This is the module dataclass for the fully-connected transition model.

    The transition probability is split into a deterministic RNN and a fully connected MLP as follows.

    .. math::

        \begin{array}{ll}
        h^t = f(h^{t-1}, z^{t-1}, a^{t-1}),
        z^t ~ \mathcal{N}(\mu(h^t), \sigma^2(h^t)),
        \end{array}
    
    where h is the hidden state of the RNN, f(.) is the RNN, mu and sigma are the mean and variance of the predicted distribution.
    See eq. 3 in the paper.

    Attributes:
        latent_dim (int): the dimension of the latent space.
        hidden_dim (int): the dimensino of hidden units in the RNN and the MLP.
    """

    latent_dim: int = 9
    hidden_dim: int = 64

    @nn.compact
    def __call__(
        self, hidden: jnp.DeviceArray, x: jnp.DeviceArray, mask: jnp.DeviceArray
    ) -> tuple[jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray]:
        """ Computes the mean and log variance of the transition probability p(z^t | z^{t-1}, a^{t-1}).

        Args:
            hidden: The previous hidden state of the RNN (h^{t-1}). 
            x: The concatentated input vector of latent state and action, [z, a].
            mask: Not used in this fully-connected case.

        Returns:
            h: The hidden state of RNN at the current timestep (h^t).
            mu: The predicted mean of the latent state distribution.
            log_var: The predicted log variance fo the latent state distribution.
        """
        # the log variance is a learnable parameter
        log_var = self.param("log_var", nn.initializers.zeros, (self.latent_dim,))
        h = nn.GRUCell()(hidden, x)[0]
        h = nn.relu(nn.Dense(features=self.hidden_dim)(h))
        h = nn.relu(nn.Dense(features=self.hidden_dim)(h))
        mu = nn.Dense(features=self.latent_dim)(h)
        # clipping log variance to -3
        log_var = jnp.maximum(log_var, -3.0)
        return h, mu, jnp.broadcast_to(log_var, (x.shape[0], log_var.shape[0]))


class ParallelRNN(nn.Module):
    r""" This is the module dataclass for the VCD transition model.

    The transition probability of each dimension is computed by a seperate network,
    and is split into a masked deterministic RNN and a MLP as follows.

    .. math::

        \begin{array}{ll}
        h^t_i = f(h^{t-1}_i, M^{\mathcal{G}}_i \odot [z^{t-1}, a^{t-1}]),
        z^t_i ~ \mathcal{N}(\mu(h^t_i), \sigma^2(h^t_i)),
        \end{array}
    
    where h is the hidden state of the RNN, f(.) is the RNN, mu and sigma are the mean and variance of the predicted distribution.
    See eq. 9 in the paper.

    Attributes:
        latent_dim (int): the dimension of the latent space.
        hidden_dim (int): the dimensino of hidden units in the RNN and the MLP.
    """

    latent_dim: int = 9
    hidden_dim: int = 64

    @nn.compact
    def __call__(
        self, hidden: jnp.DeviceArray, x: jnp.DeviceArray, mask: jnp.DeviceArray
    ) -> tuple[jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray]:
        """ Computes the mean and log variance of the transition probability p(z^t | z^{t-1}, a^{t-1}).

        Args:
            hidden: The previous hidden state of the RNN (h^{t-1}).
            x: The concatentated input vector of latent state and action, [z, a].
            mask: The binary mask according to the causal graph. 

        Returns:
            h: The hidden state of RNN at the current timestep (h^t).
            mu: The predicted mean of the latent state.
            log_var: The predicted log variance fo the latent state.
        """
        # the log variance is a learnable parameter
        log_var = self.param("log_var", nn.initializers.zeros, (self.latent_dim,))
        # Repeating the input vector along axis -1 in order to apply different masks for each latent dimension
        x = jnp.repeat(jnp.expand_dims(x, -1), self.latent_dim, -1)
        x = x * mask
        # Predictions in all dimensions are parallelised via vmap over axis -1, with no parameter sharing.
        h = nn.vmap(
            nn.GRUCell,
            in_axes=-1,
            out_axes=-1,
            variable_axes={"params": 0},
            split_rngs={"params": True},
        )()(hidden, x)[0]
        h = nn.vmap(
            nn.Dense,
            in_axes=-1,
            out_axes=-1,
            variable_axes={"params": 0},
            split_rngs={"params": True},
        )(features=self.hidden_dim)(h)
        h = nn.relu(h)
        h = nn.vmap(
            nn.Dense,
            in_axes=-1,
            out_axes=-1,
            variable_axes={"params": 0},
            split_rngs={"params": True},
        )(features=self.hidden_dim)(h)
        h = nn.relu(h)
        mu = nn.vmap(
            nn.Dense,
            in_axes=-1,
            out_axes=-1,
            variable_axes={"params": 0},
            split_rngs={"params": True},
        )(features=1)(h)
        # clipping log variance to -3
        log_var = jnp.maximum(log_var, -3.0)
        return (
            h,
            jnp.squeeze(mu, axis=-2),
            jnp.broadcast_to(log_var, (x.shape[0], log_var.shape[0])),
        )
