import jax.random as random
import jax.numpy as jnp
import jax
import flax
from flax.training import train_state
import flax.linen as nn
import optax
from typing import Callable, Union, Tuple


def KL_div(
    mu_q: jnp.DeviceArray,
    logvar_q: jnp.DeviceArray,
    mu_p: jnp.DeviceArray,
    logvar_p: jnp.DeviceArray,
) -> jnp.DeviceArray:
    """ Returns the KL divergence KL[Q || P] between two Gaussian distributions.

    Args:
        mu_q (DeviceArray): The mean vector of the distribution Q.
        logvar_q (DeviceArray): A vector containing the log variance of Q for each dimension.
        mu_p (DeviceArray): The mean vector of the distribution P.
        logvar_p (DeviceArray): A vector containing the log variance of P for each dimension.

    Returns:
        kl (DeviceArray): A vector where each entry is the KL divergence of the corresponding dimension.
    """
    kl = 0.5 * (
        logvar_p
        - logvar_q
        + (jnp.exp(logvar_q) + (mu_q - mu_p) ** 2) / (jnp.exp(logvar_p))
        - 1
    )
    return kl


class BaseSequenceModel(nn.Module):
    """ The base class for ELBO-based latent sequence models.
    
    To be implemented:
        setup: A function to specify the encoders, decoders and the transition models.
        sparsity: Returns the expected number of edges in the learnt causal graph (if applicable).
        intervention_sparsity: Returns the expected number of intervention targets (if applicable).
        get_init_carry: Returns the initial carry dict, containing the relevant hidden state, masks and prior probabilities.


    Attributes:
        latent_dim {int}: The dimension of the latent space.
        action_dim {int}: The dimension of the action space.
        hidden_dim {int}: The dimension of hidden units in the transition model 
            (and observation model in the case of MLP observations).
        obs_dim (Union[int, None]): The dimension of the observation space (needed in the Mixed-state case only).
        n_env {int}: The number for intervened environments plus one (for the undisturbed environment).    
    """

    latent_dim: int
    action_dim: int
    hidden_dim: int
    obs_dim: Union[int, None]
    n_env: int

    def setup(self) -> None:
        """ To be implemented for specific world models.

        Requires the implementation of the following functions:
        prior: The transition model. h^{t-1}, z^{t-1}, a^{t-1}, mask -> h^t, mu_{z^t}, logvar_{z^t}
        int_prior: The transition model for intervened environments. Used in VCD only. 
            h^{t-1}, z^{t-1}, a^{t-1}, mask -> h^t, mu_{z^t}, logvar_{z^t}
        posterior: observation -> mu_{z^t}, logvar_{z^t}
        obs_model: z^t -> observation
        """
        self.prior: Callable[
            [jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray],
            Tuple[(jnp.DeviceArray,) * 3],
        ]
        self.int_prior: Callable[
            [jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray],
            Tuple[(jnp.DeviceArray,) * 3],
        ]
        self.posterior: Callable[
            [jnp.DeviceArray], tuple[jnp.DeviceArray, jnp.DeviceArray]
        ]
        self.obs_model: Callable[[jnp.DeviceArray], jnp.DeviceArray]
        raise NotImplementedError

    @staticmethod
    def reparameterize(
        rng: jnp.DeviceArray, mean: jnp.DeviceArray, logvar: jnp.DeviceArray
    ) -> jnp.DeviceArray:
        """ Returns a sample from a Gaussian distribution using the reparameterization trick.
        
        Args:
            rng (random.PRNGKey): The random key for the sample.
            mean (DeviceArray): The mean of the Gaussian distribution from which the samples are drawn.
            logvar (DeviceArray): The log variance of the Gaussian distribution from which the samples are drawn.

        Returns:
            DeviceArray: The random samples. If batched, the returned samples have the same batch size as logvar.
        """
        std = jnp.exp(0.5 * logvar)
        eps = random.normal(rng, logvar.shape)
        return mean + eps * std

    @staticmethod
    def sparsity(params: flax.core.frozen_dict.FrozenDict) -> jnp.DeviceArray:
        """ Computes the expected number of causal edges in the learnt causal graph.
        
        Args:
            params (FrozenDict): The parameter dict for the model.

        Returns:
            The expected number of edges in the causal graph.
        """
        raise NotImplementedError

    @staticmethod
    def intervention_sparsity(
        params: flax.core.frozen_dict.FrozenDict,
    ) -> jnp.DeviceArray:
        """ Computes the expected number of intervention targets.
        
        Args:
            params (FrozenDict): The parameter dict for the model.

        Returns:
            The expected number of intervention targets.
        """
        raise NotImplementedError

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
            batch (DeviceArray): A batch of observations. (batch_size x n_envs x *observation dimensions)
            params (FrozenDict): The parameter dict containing the causal graph mask and intervention target mask.
            rng (DeviceArray): The rng key used for sampling graphs and intervention targets.
        
        Returns:
            A dictionary containing the relevant information carried from one timestep to the next, i.e.
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
        raise NotImplementedError

    def encode(self, obs):
        return self.posterior(obs)

    def decode(self, z):
        return self.obs_model(z)

    def predict_next(self, hidden, z, action, mask):
        return self.prior(hidden, z, action, mask)

    def __call__(
        self, prev_step: dict, obs: jnp.DeviceArray, action: jnp.DeviceArray
    ) -> tuple[dict, Tuple[(jnp.DeviceArray,) * 3]]:
        """ Computes the reconstruction error and the KL divergence for the next timestep.
        
        Args:
            prev_step (dict): A dict containing infromation from the previous timestep.
            obs (DeviceArray): The observation for the current timestep.
            action (DeviceArray): The observed action for the current timestep.

        Returns:
            carry (dict): A dict containing the relevant information to be carried to the next timestep.
            tuple:
                recon (DeviceArray): The reconstructed observation.
                kl (DeviceArray): The KL divergence between the prior and posterior for each dimension (not summed).
                latent_error (DeviceArray): The squared error between the mean of the prediction and 
                    the posterior (for logging purpose only).
        """
        rng = prev_step["rng"]
        t_mask = prev_step["transition_mask"]
        int_mask = prev_step["int_mask"]
        int_mean_p = prev_step["int_prior_mu"]
        int_logvar_p = prev_step["int_prior_logvar"]
        mean_p = prev_step["prior_mu"]
        logvar_p = prev_step["prior_logvar"]

        # Encode the observation and sample from the posterior
        mean_q, logvar_q = self.posterior(obs)
        z = self.reparameterize(rng, mean_q, logvar_q)

        # Compute the reconstruction and the KL divergence between the prior and the posterior
        recon = self.obs_model(z)
        obs_kl = KL_div(mean_q, logvar_q, mean_p, logvar_p)
        int_kl = KL_div(mean_q, logvar_q, int_mean_p, int_logvar_p)

        # the KL term can be computed by adding the masked kl from the interventional model and the undisturbed model. See eq. 24.
        kl = int_mask * int_kl + (1 - int_mask) * obs_kl
        latent_error = (
            int_mask * (int_mean_p - mean_q) ** 2
            + (1 - int_mask) * (mean_p - mean_q) ** 2
        )

        # Predict forward.
        h_t, next_mu, next_logvar = self.prior(prev_step["hidden"], z, action, t_mask)
        int_h_t, int_next_mu, int_next_logvar = self.int_prior(
            prev_step["int_hidden"], z, action, t_mask
        )

        # update the random key to ensure different reparameterisation samples for next timestep
        rng, key = random.split(rng)
        carry = {
            "hidden": h_t,
            "prior_mu": next_mu,
            "prior_logvar": next_logvar,
            "int_hidden": int_h_t,
            "int_prior_mu": int_next_mu,
            "int_prior_logvar": int_next_logvar,
            "action": action,
            "transition_mask": t_mask,
            "int_mask": int_mask,
            "rng": rng,
        }
        return carry, (recon, kl, latent_error)

    def rollout(
        self,
        prev_step: dict,
        obs: Union[None, jnp.DeviceArray],
        action: jnp.DeviceArray,
    ):
        """ Rollout the next timestep with or without observations.

        If obs is available, the posterior distribution is used.
        If obs is none, the prior from the transition model is used instead.
        In both cases, the returned reconstruction is from the predicted latents for evaluation purpose. 
        When observation is available, this corresponds to one-step prediction in the observation space. 
        
        Args:
            prev_step (dict): A dict containing infromation from the previous timestep.
            obs (Union[None, DeviceArray]): The observation at current timestep. Set to None if not available.
            action (DeviceArray): The action for the current timestep.

        Returns:
            carry (dict): A dict containing the relevant information to be carried to the next timestep.
            tuple:
                recon (DeviceArray): The predicted reconstruction of the observation.
                latent_error (DeviceArray): The squared error in the latent space.
        """
        rng = prev_step["rng"]
        t_mask = prev_step["transition_mask"]
        int_mask = prev_step["int_mask"]
        int_mean_p = prev_step["int_prior_mu"]
        mean_p = prev_step["prior_mu"]

        # Reconstruct using prior distribution from the transition model.
        mean_prior = int_mask * int_mean_p + (1 - int_mask) * mean_p
        recon = self.obs_model(z)
        if obs is not None:
            # Use the posterior if observation is available.
            mean_q, _ = self.posterior(obs)
            latent_error = (mean_prior - mean_q) ** 2
            z = mean_q
        else:
            # Use the prior if there is no observation.
            latent_error = None
            z = mean_prior

        # Predict the next timestep based on the estimated latent for the current timestep.
        h_t, next_mu, next_logvar = self.prior(prev_step["hidden"], z, action, t_mask)
        int_h_t, int_next_mu, int_next_logvar = self.int_prior(
            prev_step["int_hidden"], z, action, t_mask
        )

        rng, key = random.split(rng)
        carry = {
            "hidden": h_t,
            "prior_mu": next_mu,
            "prior_logvar": next_logvar,
            "int_hidden": int_h_t,
            "int_prior_mu": int_next_mu,
            "int_prior_logvar": int_next_logvar,
            "action": action,
            "transition_mask": t_mask,  # Note that the masks are not re-sampled within a trajectory.
            "int_mask": int_mask,
            "rng": rng,
        }
        return carry, (recon, latent_error)

    def init_train_state(
        self,
        rng: jnp.DeviceArray,
        batch: Tuple[jnp.DeviceArray, jnp.DeviceArray],
        lr: float = 0.001,
    ) -> train_state.TrainState:
        """ Returns the initial TrainState of the model.

        Args:
            rng (DeviceArray): A random key to initialise parameters.
            batch (Tuple[DeviceArray, DeviceArray]): A batch of (observation, action) sequence. 
                The sizes are (timesteps x batch x interventions x *observation_dim/ *action_dim)
            lr (float): The learning rate.
    
        Return:
            The initial TrainState.
        """
        batched_obs = batch[0][0]
        mask = {
            "causal_graph": jnp.ones(
                (self.latent_dim + self.action_dim, self.latent_dim)
            ),
            "intervention_targets": jnp.ones((self.n_env, self.latent_dim)),
        }
        init_carry = self.get_init_carry(
            self.hidden_dim, self.latent_dim, self.action_dim, batched_obs, mask, rng
        )
        params = self.init(rng, init_carry, batch[0][0], batch[1][0], batch[2][0])
        tx = optax.adam(learning_rate=lr)
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=tx)

    def load_train_state(
        self,
        rng: jnp.DeviceArray,
        batch: Tuple[jnp.DeviceArray, jnp.DeviceArray],
        pretrained_encoder: dict,
        pretrained_decoder: dict,
        lr: float = 0.0001,
    ) -> train_state.TrainState:
        """ Returns the initial TrainState of the model with pretrained encoders and decoders.

        Args:
            rng (DeviceArray): A random key to initialise parameters.
            batch (Tuple[DeviceArray, DeviceArray]): A batch of (observation, action) sequence. 
                The sizes are (timesteps x batch x interventions x *observation_dim/ *action_dim)
            pretrained_encoder (dict): A dict containing the pretrained parameters for the encoder.
            pretrained_decoder (dict): A dict containing the pretrained parameters for the decoder.
            lr (float): The learning rate.
    
        Return:
            The initial TrainState.
        """
        batched_obs = batch[0][0]
        mask = {
            "transition_mask": jnp.ones(
                (self.latent_dim + self.action_dim, self.latent_dim)
            ),
            "policy_mask": jnp.ones((self.latent_dim, self.action_dim)),
            "reward_mask": jnp.ones((self.latent_dim + self.action_dim, 1)),
            "intervention_mask": jnp.ones((self.n_env, self.latent_dim)),
        }
        init_carry = self.get_init_carry(
            self.hidden_dim, self.latent_dim, self.action_dim, batched_obs, mask, rng
        )
        params = flax.core.unfreeze(
            self.init(rng, init_carry, batch[0][0], batch[1][0], batch[2][0])
        )
        params["params"]["obs_net"] = pretrained_decoder
        params["params"]["posterior_net"] = pretrained_encoder
        params = flax.core.freeze(params)
        tx = optax.adam(learning_rate=lr)
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=tx)

    @classmethod
    @jax.tree_util.Partial(jax.jit, static_argnums=(0, 5))
    def train(
        cls,
        state: train_state.TrainState,
        data: Tuple[jnp.DeviceArray, jnp.DeviceArray],
        rng: jnp.DeviceArray,
        lambdas: dict,
        dimensions: Tuple[(int,) * 3] = (64, 24, 2),
    ) -> Tuple[Tuple[(jnp.DeviceArray,) * 5], train_state.TrainState]:
        """ Perform a gradient step on the parameters of the model.
        
        Args:
            state (TrainState): The train state of the model.
            data (tuple[DeviceArray, DeviceArray]): The observation action sequence. 
                The dimensions are [batch, timestep, env_id, observation_dim/ action_dim] 
            rng (DeviceArray): The initial random key.
            lambdas (Dict): A dict containing the hyperparameters.
                {
                lambdas['kl'] (float)
                lambdas['sparse'] (float)
                lambdas['int'] (float)
                }
            dimensions (tuple[int, int, int]): The dimensions of the latent space, 
                hidden units and the action space (latent_dim, hidden_dim, action_dim).

        Return:
            loss (tuple): The individual terms of the loss value 
                (reconstruction loss, KL, prediction error, sparsity loss, intervention sparsity loss).
            state (TrainState): The updated TrainState.
        """

        def loss_fn(params):
            init_carry = cls.get_init_carry(
                *dimensions, data[0][0], params["params"], rng
            )
            _, outputs = jax.lax.scan(
                lambda c, x: state.apply_fn(params, c, x[0], x[1]), init_carry, data
            )
            recon = outputs[0]
            recon_mse = ((recon - data[0]) ** 2).mean(axis=(0, 1, 2)).sum()
            kl = outputs[1].mean(axis=(0, 1, 2)).sum()
            l_error = outputs[2].mean(axis=1)
            sparse_loss = cls.sparsity(params)
            int_loss = cls.intervention_sparsity(params)
            return (
                recon_mse
                + lambdas["kl"] * kl
                + lambdas["sparse"] * sparse_loss
                + lambdas["int"] * int_loss,
                (recon_mse, kl, l_error, sparse_loss, int_loss,),
            )

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return loss[1], state

    @classmethod
    @jax.tree_util.Partial(jax.jit, static_argnums=(0, 5))
    def evaluate(
        cls,
        state: train_state.TrainState,
        data: Tuple[jnp.DeviceArray, jnp.DeviceArray],
        rng: jnp.DeviceArray,
        lambdas: dict,
        dimensions: Tuple[(int,) * 3] = (64, 24, 2),
    ) -> Tuple[Tuple[(jnp.DeviceArray,) * 5], train_state.TrainState]:
        """ Evaluate the loss given a batch of data and the parameters.
        
        Args:
            state (TrainState): The train state of the model.
            data (tuple[DeviceArray, DeviceArray]): The observation action sequence. 
                The dimensions are [batch, timestep, env_id, observation_dim/ action_dim] 
            rng (DeviceArray): The initial random key.
            lambdas (Dict): A dict containing the hyperparameters.
                {
                lambdas['kl'] (float)
                lambdas['sparse'] (float)
                lambdas['int'] (float)
                }
            dimensions (tuple[int, int, int]): The dimensions of the latent space, 
                hidden units and the action space (latent_dim, hidden_dim, action_dim).

        Return:
            loss (tuple): The individual terms of the loss value 
                (reconstruction loss, KL, prediction error, sparsity loss, intervention sparsity loss).
        """

        def loss_fn(params):
            init_carry = cls.get_init_carry(
                *dimensions, data[0][0], params["params"], rng
            )
            _, outputs = jax.lax.scan(
                lambda c, x: state.apply_fn(params, c, x[0], x[1]), init_carry, data
            )
            recon = outputs[0]
            recon_mse = ((recon - data[0]) ** 2).mean(axis=(0, 1, 2)).sum()
            kl = outputs[1].mean(axis=(0, 1, 2)).sum()
            l_error = outputs[2].mean(axis=1)
            sparse_loss = cls.sparsity(params)
            int_loss = cls.intervention_sparsity(params)
            return (
                recon_mse
                + lambdas["kl"] * kl
                + lambdas["sparse"] * sparse_loss
                + lambdas["int"] * int_loss,
                (recon_mse, kl, l_error, sparse_loss, int_loss,),
            )

        return loss_fn(state.params)[1]
