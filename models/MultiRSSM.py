from models.RSSM import RSSM
from modules.mlp import MLP
from modules.decoders import Decoder
from modules.encoders import Encoder
from modules.transitions import TransitionRNN
import flax.linen as nn
import jax.numpy as jnp


class MultiRSSM(RSSM):
    """ This is the MultiRSSM model class for the mixed-state experiment.

    Same as RSSM except the transition model parameters are not shared across
    environments.

    Attributes:
        latent_dim {int}: The dimension of the latent space.
        action_dim {int}: The dimension of the action space.
        hidden_dim {int}: The dimension of hidden units in the transition model
            (and observation model in the case of MLP observations).
        obs_dim (Union[int, None]): The dimension of the observation space
            (needed in the Mixed-state case only).
        n_env {int}: The number for intervened environments plus one
            (for the undisturbed environment).
    """

    latent_dim: int = 24
    obs_dim: int = 12
    action_dim: int = 2
    hidden_dim: int = 64
    n_env: int = 6

    def setup(self):
        """ The transition model is implemented as an RNN. The encoders and decoders
        are implemented as MLPs.

        The encoders and decoders are shared across all environments
        but seperate transition models are used for each individual environment.

        prior: The transition model. (h, z,a)^{t-1}, mask -> (h, mu_z, logvar_z)^t
        int_prior: The transition model for intervened environments. Used in VCD only.
        posterior: observation -> mu_{z^t}, logvar_{z^t}
        obs_model: z^t -> observation
        """
        self.posterior_net = nn.vmap(
            MLP,
            in_axes=(1, None),
            out_axes=1,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )(out_dim=self.latent_dim, hidden_dim=self.hidden_dim)

        self.prior_net = nn.vmap(
            TransitionRNN,
            in_axes=(1, 1, None),
            out_axes=1,
            variable_axes={"params": 0},  # Note that the parameters are not shared.
            split_rngs={"params": True},
        )(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim)
        self.obs_net = nn.vmap(
            MLP,
            in_axes=(1, None),
            out_axes=1,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )(out_dim=self.obs_dim, hidden_dim=self.hidden_dim)

        self.prior = lambda h, z, a, mask: self.prior_net(
            h, jnp.concatenate([z, a], axis=-1), 1
        )
        # int_prior is not used in this model.
        self.int_prior = lambda h, z, a, mask: (0, 0, 0)
        self.obs_model = lambda z: self.obs_net(z, 1)[0]
        self.posterior = lambda obs: self.posterior_net(obs, 1)


class ImageMultiRSSM(MultiRSSM):
    """ This is the MultiRSSM model class for the image experiment.

    Same as RSSM except the transition model parameters are not shared across
    environments.

    Attributes:
        latent_dim {int}: The dimension of the latent space.
        action_dim {int}: The dimension of the action space.
        hidden_dim {int}: The dimension of hidden units in the transition model
            (and observation model in the case of MLP observations).
        obs_dim (Union[int, None]): The dimension of the observation space
        (needed in the Mixed-state case only).
        n_env {int}: The number for intervened environments plus one
        (for the undisturbed environment).
    """

    latent_dim: int = 24
    obs_dim: int = 12
    action_dim: int = 2
    hidden_dim: int = 64
    n_env: int = 6

    def setup(self):
        """ The transition model is implemented as an RNN. The encoders and decoders
        are implemented as CNNs.

        The encoders and decoders are shared across all environments but
        seperate transition models are used for each individual environment.

        prior: The transition model. (h, z,a)^{t-1}, mask -> (h, mu_z, logvar_z)^t
        int_prior: The transition model for intervened environments. Used in VCD only.
        posterior: observation -> mu_{z^t}, logvar_{z^t}
        obs_model: z^t -> observation
        """
        self.posterior_net = nn.vmap(
            Encoder,
            in_axes=1,
            out_axes=1,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )(latent_dim=self.latent_dim)
        self.prior_net = nn.vmap(
            TransitionRNN,
            in_axes=(1, 1, None),
            out_axes=1,
            variable_axes={"params": 0},  # Note that the parameters are not shared.
            split_rngs={"params": True},
        )(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim)
        self.obs_net = nn.vmap(
            Decoder,
            in_axes=1,
            out_axes=1,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )()

        self.prior = lambda h, z, a, mask: self.prior_net(
            h, jnp.concatenate([z, a], axis=-1), 1
        )
        # int_prior is not used in this model.
        self.int_prior = lambda h, z, a, mask: (0, 0, 0)
        self.obs_model = lambda z: self.obs_net(z)
        self.posterior = lambda obs: self.posterior_net(obs)
