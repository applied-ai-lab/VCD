from models.sequence_model import BaseSequenceModel
from jax import numpy as jnp
from jax import random
import flax
import flax.linen as nn
from modules.mlp import MLP
from modules.transitions import TransitionRNN
from modules.encoders import Encoder
from modules.decoders import Decoder
from typing import Union


class RSSM(BaseSequenceModel):
    """ This is the RSSM model class for the mixed-state experiment.

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

    def setup(self) -> None:
        """ The transition model is implemented as an RNN. The encoders and decoders
        are implemented as MLPs.

        All models are vmap'ed over the interventions dimension with shared parameters.

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
            variable_axes={"params": None},
            split_rngs={"params": False},
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

    @classmethod
    def get_init_carry(
        cls,
        hidden_dim: int,
        latent_dim: int,
        action_dim: int,
        batch: jnp.DeviceArray,
        params: Union[dict, flax.core.frozen_dict.FrozenDict],
        rng: jnp.DeviceArray,
    ) -> dict:
        """ Returns the initial carry dict.

        Args:
            hidden_dim (int): The dimension of the hidden units in the RNN and MLP.
            latent_dim (int): The dimension of the latent space.
            action_dim (int): The dimension of the action space.
            batch (DeviceArray): A batch of observations.
                (batch_size x n_envs x *observation dimensions)
            params (FrozenDict): The parameter dict containing the causal graph mask
                and intervention target mask.
            rng (DeviceArray): The rng key used for sampling graphs and intervention
                targets.

        Returns:
            A dictionary containing the relevant information carried from one timestep
            to the next, i.e.
            {
                "hidden": ...,
                "prior_mu": ...,
                "prior_logvar": ...,
                "int_hidden": ...,
                "int_prior_mu": ...,
                "int_prior_logvar": ...,
                "action": ...,
                "transition_mask": ...,
                "int_mask": ...,
                "rng": ...,
            }
        """

        # Initialise the hidden state and initial action to zeros.
        h_0 = jnp.zeros((batch.shape[0], batch.shape[1], hidden_dim))
        a_0 = jnp.zeros((batch.shape[0], batch.shape[1], action_dim))

        # The initial prior distribution is unit normal.
        mu, logvar = (
            jnp.zeros((batch.shape[0], batch.shape[1], latent_dim)),
            jnp.zeros((batch.shape[0], batch.shape[1], latent_dim)),
        )
        rng, key = random.split(rng)

        # The int_... fields and the masks are not used in RSSM.
        carry = {
            "hidden": h_0,
            "prior_mu": mu,
            "prior_logvar": logvar,
            "int_hidden": 0,
            "int_prior_mu": 0,
            "int_prior_logvar": 0,
            "action": a_0,
            "transition_mask": 0,
            "int_mask": 0,
            "rng": rng,
        }
        return carry

    @staticmethod
    def sparsity(params):
        """ Returns 0 as there is no sparsity loss in RSSM."""
        return 0

    @staticmethod
    def intervention_sparsity(params):
        """ Returns 0 as there is no sparsity loss in RSSM."""
        return 0


class ImageRSSM(RSSM):
    """ This is the RSSM model class for the image experiment.

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
    obs_dim: int = None
    action_dim: int = 2
    hidden_dim: int = 64
    n_env: int = 6

    def setup(self):
        """ The transition model is implemented as an RNN. The encoders and decoders
        are implemented as CNN/ deconvolution network.

        All models are vmap'ed over the interventions dimension with shared parameters.

        prior: The transition model. (h, z,a)^{t-1}, mask -> (h, mu_z, logvar_z)^t
        int_prior: The transition model for intervened environments. Used in VCD only.
        posterior: observation -> mu_{z^t}, logvar_{z^t}
        obs_model: z^t -> observation
        """
        self.reward_net = nn.vmap(
            MLP,
            in_axes=(1, None),
            out_axes=1,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )(out_dim=1, hidden_dim=self.hidden_dim)
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
            variable_axes={"params": None},
            split_rngs={"params": False},
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
