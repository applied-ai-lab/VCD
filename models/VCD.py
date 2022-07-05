from models.sequence_model import BaseSequenceModel
from modules.mlp import MLP
from modules.decoders import Decoder
from modules.encoders import Encoder
from modules.transitions import ParallelRNN
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from typing import Union


class VCD(BaseSequenceModel):
    """This is the VCD model class for the mixed-state experiment. See eq. 8 for details.

    Attributes:
        latent_dim {int}: The dimension of the latent space.
        action_dim {int}: The dimension of the action space.
        hidden_dim {int}: The dimension of hidden units in the transition model
            (and observation model in the case of MLP observations).
        obs_dim (Union[int, None]): The dimension of the observation space (needed in
            the Mixed-state case only).
        n_env {int}: The number for intervened environments plus one (for the
            undisturbed environment).
    """

    latent_dim: int = 16
    obs_dim: int = 12
    action_dim: int = 2
    hidden_dim: int = 64
    n_env: int = 6

    def setup(self):
        """The transition model is implemented as RNNs for each individual dimension.
        The encoders and decoders are implemented as MLPs.

        The encoders and decoders are shared across all environments.
        There are two transition models, one for the undisturbed transition mechanism
        and one for the intervened mechanisms.
        Each environment has a seperate intervened transition mechanism whereas the
        undisturbed mechanism is shared across all environments.

        The VCD model also has a probilistic belief over graph structures and
        intervention targets.
        The probability of each edge {i,j} can be computed as exp(causal_graph[i,j]).

        prior: The transition model. (h, z,a)^{t-1}, mask -> (h, mu_z, logvar_z)^t
        int_prior: The transition model for intervened environments. Used in VCD only.
        posterior: observation -> mu_{z^t}, logvar_{z^t}
        obs_model: z^t -> observation
        """
        self.prior_net = nn.vmap(
            ParallelRNN,
            in_axes=(1, 1, 1),
            out_axes=1,
            variable_axes={
                "params": None
            },  # Parameters are shared across environment for the observed mechanisms.
            split_rngs={"params": False},
        )(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim)
        self.int_prior_net = nn.vmap(
            ParallelRNN,
            in_axes=(1, 1, 1),
            out_axes=1,
            variable_axes={
                "params": 0
            },  # Parameters are not shared for the intervened mechanisms.
            split_rngs={"params": True},
        )(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim)
        self.posterior_net = nn.vmap(
            MLP,
            in_axes=(1, None),
            out_axes=1,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )(out_dim=self.latent_dim, hidden_dim=self.hidden_dim)
        self.obs_net = nn.vmap(
            MLP,
            in_axes=(1, None),
            out_axes=1,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )(out_dim=self.obs_dim, hidden_dim=self.hidden_dim)

        self.prior = lambda h, z, a, mask: self.prior_net(
            h, jnp.concatenate([z, a], axis=-1), mask
        )
        self.int_prior = lambda h, z, a, mask: self.int_prior_net(
            h, jnp.concatenate([z, a], axis=-1), mask
        )
        self.obs_model = lambda z: self.obs_net(z, 1)[0]
        self.posterior = lambda obs: self.posterior_net(obs, 1)

        # Probabilistic belief over graph structures and intervention targets.
        self.causal_graph = self.param(
            "causal_graph",
            lambda *x: 4 * nn.initializers.ones(*x),
            (self.latent_dim + self.action_dim, self.latent_dim),
        )
        self.intervention_targets = self.param(
            "intervention_targets",
            # The intervention target for dimension -1 is set to -1e8 to make sure that
            # the undisturbed environment has no interventions.
            lambda *x: jnp.concatenate(
                [5 * nn.initializers.ones(*x), -1e8 * jnp.ones((1, self.latent_dim))],
                axis=-2,
            ),
            (self.n_env - 1, self.latent_dim),
        )

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
        """Returns the initial carry dict.

        Args:
            hidden_dim (int): The dimension of the hidden units in the RNN and MLP.
            latent_dim (int): The dimension of the latent space.
            action_dim (int): The dimension of the action space.
            batch (DeviceArray): A batch of observations.
                The dimensions are (batch_size x n_envs x *observation dimensions)
            params (FrozenDict): The parameter dict containing the causal graph mask
                and intervention target mask.
            rng (DeviceArray): The rng key used for sampling graphs and interventions.

        Returns:
            A dictionary containing the relevant information carried to the next
            timestep:
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
        h_0 = jnp.zeros((batch.shape[0], batch.shape[1], hidden_dim, latent_dim))
        a_0 = jnp.zeros((batch.shape[0], batch.shape[1], action_dim))

        # Draw samples of the intervention targets and causal graphs using the
        # Gumbel-max trick.
        rng1, rng2 = random.split(rng)
        trans_mask = cls.gumbel_max_state(params["causal_graph"], batch, rng1)
        int_mask = cls.gumbel_max_intervention(
            params["intervention_targets"], batch, rng2
        )

        # The initial prior distribution is unit normal.
        mu, logvar = (
            jnp.zeros((batch.shape[0], batch.shape[1], latent_dim)),
            jnp.zeros((batch.shape[0], batch.shape[1], latent_dim)),
        )
        rng, key = random.split(rng2)
        carry = {
            "hidden": h_0,
            "prior_mu": mu,
            "prior_logvar": logvar,
            "int_hidden": h_0,
            "int_prior_mu": mu,
            "int_prior_logvar": logvar,
            "action": a_0,
            "transition_mask": trans_mask,
            "int_mask": int_mask,
            "rng": rng,
        }
        return carry

    @staticmethod
    def gumbel_max_state(
        log_prob: jnp.DeviceArray, input: jnp.DeviceArray, rng: jnp.DeviceArray
    ) -> jnp.DeviceArray:
        """A function to draw samples of causal graphs using the Gumbel-max trick.

        Args:
            log_prob (DeviceArray): The log probability of each edge.
            input (DeviceArray): A batch of latent states. Used for determining
                the batch size only.
            rng (DeviceArray): A random seed.

        Returns:
            A DeviceArray with the sampled adjacency matrices, with the same
            batch size as input.
        """
        shape = (input.shape[0], input.shape[1], log_prob.shape[0], log_prob.shape[1])
        logistic_samples = random.logistic(rng, shape)
        log_prob = jnp.broadcast_to(log_prob, shape) + logistic_samples
        return (
            jnp.float32(log_prob > 0)
            + jax.nn.sigmoid(log_prob)
            - jax.lax.stop_gradient(jax.nn.sigmoid(log_prob))
        )

    @staticmethod
    def gumbel_max_intervention(
        log_prob: jnp.DeviceArray, input: jnp.DeviceArray, rng: jnp.DeviceArray
    ) -> jnp.DeviceArray:
        """A function to draw samples of intervention targets using the Gumbel-max trick.

        Args:
            log_prob (DeviceArray): The log probability of each intervention target.
            input (DeviceArray): A batch of latent states. Used for determining
                the batch size only.
            rng (DeviceArray): A random seed.

        Returns:
            A DeviceArray with the sampled binary intervention target matrix, with
            the same batch size as input.
        """
        shape = (input.shape[0], log_prob.shape[0], log_prob.shape[1])
        logistic_samples = random.logistic(rng, shape)
        log_prob = jnp.broadcast_to(log_prob, shape) + logistic_samples
        return (
            jnp.float32(log_prob > 0)
            + jax.nn.sigmoid(log_prob)
            - jax.lax.stop_gradient(jax.nn.sigmoid(log_prob))
        )

    @staticmethod
    def sparsity(params):
        """Returns the expected number of edges in the causal graph."""
        return jnp.sum(jax.nn.sigmoid(params["params"]["causal_graph"]))

    @staticmethod
    def intervention_sparsity(params):
        """Returns the expected number of intervention targets"""
        return jnp.sum(jax.nn.sigmoid(params["params"]["intervention_targets"]))


class ImageVCD(VCD):
    """This is the VCD model class for the image experiment. See eq. 8 for details.

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

    latent_dim: int = 16
    obs_dim: int = 12
    action_dim: int = 2
    hidden_dim: int = 64
    n_env: int = 6

    def setup(self):
        """The transition model is implemented as RNNs for each individual dimension.
        The encoders and decoders are implemented as Cnns.

        The encoders and decoders are shared across all environments.
        There are two transition models, one for the undisturbed transition mechanism
        and one for the intervened mechanisms.
        Each environment has a seperate intervened transition mechanism whereas
        the undisturbed mechanism is shared across all environments.

        The VCD model also has a probilistic belief over graph structures and
        intervention targets. The probability of each edge {i,j} can be computed
        as exp(causal_graph[i,j]).

        prior: The transition model. (h, z,a)^{t-1}, mask -> (h, mu_z, logvar_z)^t
        int_prior: The transition model for intervened environments. Used in VCD only.
        posterior: observation -> mu_{z^t}, logvar_{z^t}
        obs_model: z^t -> observation
        """
        self.prior_net = nn.vmap(
            ParallelRNN,
            in_axes=(1, 1, 1),
            out_axes=1,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim)
        self.int_prior_net = nn.vmap(
            ParallelRNN,
            in_axes=(1, 1, 1),
            out_axes=1,
            variable_axes={"params": 0},
            split_rngs={"params": True},
        )(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim)
        self.posterior_net = nn.vmap(
            Encoder,
            in_axes=1,
            out_axes=1,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )(latent_dim=self.latent_dim)
        self.obs_net = nn.vmap(
            Decoder,
            in_axes=1,
            out_axes=1,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )()

        self.prior = lambda h, z, a, mask: self.prior_net(
            h, jnp.concatenate([z, a], axis=-1), mask
        )
        self.int_prior = lambda h, z, a, mask: self.int_prior_net(
            h, jnp.concatenate([z, a], axis=-1), mask
        )
        self.obs_model = lambda z: self.obs_net(z)
        self.posterior = lambda obs: self.posterior_net(obs)

        # Probabilistic belief over graph structures and intervention targets.
        self.causal_graph = self.param(
            "causal_graph",
            lambda *x: 4 * nn.initializers.ones(*x),
            (self.latent_dim + self.action_dim, self.latent_dim),
        )
        self.intervention_targets = self.param(
            "intervention_targets",
            # The intervention target for dimension -1 is set to -1e8 to make sure that
            # the undisturbed environment has no interventions.
            lambda *x: jnp.concatenate(
                [5 * nn.initializers.ones(*x), -1e8 * jnp.ones((1, self.latent_dim))],
                axis=-2,
            ),
            (self.n_env - 1, self.latent_dim),
        )
