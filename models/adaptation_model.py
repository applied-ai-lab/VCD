import flax
import flax.linen as nn
from flax.training import train_state
from models.sequence_model import BaseSequenceModel, KL_div
import jax
import jax.numpy as jnp
import jax.random as random
import optax
from typing import Union, Tuple
from models.RSSM import RSSM
from models.MultiRSSM import MultiRSSM
from models.VCD import VCD
from modules.transitions import TransitionRNN, ParallelRNN


class BaseAdapt(BaseSequenceModel):
    trained_model: BaseSequenceModel

    def setup(self):
        raise NotImplementedError

    def __call__(
        self,
        trained_params: flax.core.frozen_dict.FrozenDict,
        prev_step: dict,
        obs: jnp.DeviceArray,
        action: jnp.DeviceArray,
    ) -> tuple[dict, Tuple[(jnp.DeviceArray,) * 3]]:
        rng = prev_step["rng"]
        t_mask = prev_step["transition_mask"]
        int_mask = prev_step["int_mask"]
        int_mean_p = prev_step["int_prior_mu"]
        int_logvar_p = prev_step["int_prior_logvar"]
        mean_p = prev_step["prior_mu"]
        logvar_p = prev_step["prior_logvar"]

        # encode and decode the observation using the trained model
        mean_q, logvar_q = self.trained_model.apply(
            trained_params, obs, method=self.trained_model.encode
        )
        # print('mean_q',mean_q.shape)
        z = self.trained_model.reparameterize(rng, mean_q, logvar_q)
        recon = self.trained_model.apply(
            trained_params, z, method=self.trained_model.decode
        )
        # Compute the KL div between prior (predicted by the adaptation model) and
        # posterior

        obs_kl = KL_div(mean_q, logvar_q, mean_p, logvar_p)
        # print('obs_kl', obs_kl.shape)
        int_kl = KL_div(mean_q, logvar_q, int_mean_p, int_logvar_p)
        # print('int_kl', int_kl.shape)

        # Compute the overall KL term under the full intervention model (eq. 24)
        kl = int_mask * int_kl + (1 - int_mask) * obs_kl
        latent_error = (
            int_mask * (int_mean_p - mean_q) ** 2
            + (1 - int_mask) * (mean_p - mean_q) ** 2
        )

        # Predict forward using the adaptation model for environment specific mechanisms
        # The trained model is used otherwise.
        h_t, next_mu, next_logvar = self.trained_model.apply(
            trained_params,
            prev_step["hidden"],
            z,
            action,
            t_mask,
            method=self.trained_model.predict_next,
        )
        int_h_t, int_next_mu, int_next_logvar = self.int_prior(
            prev_step["int_hidden"], z, action, t_mask
        )

        # update the random key to ensure different reparameterisation samples for next
        # timestep
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

    @classmethod
    def get_init_carry(
        cls,
        trained_params: flax.core.FrozenDict,
        hidden_dim: int,
        latent_dim: int,
        action_dim: int,
        batch: jnp.DeviceArray,
        params: Union[dict, flax.core.frozen_dict.FrozenDict],
        rng: jnp.DeviceArray,
    ) -> dict:
        """ Returns the initial carry dict.

        Args:
            trained_params (FrozenDict): A dict containing the parameters of the
                frozen trained model.
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
        raise NotImplementedError

    def rollout(
        self,
        trained_params: flax.core.FrozenDict,
        prev_step: dict,
        obs: Union[None, jnp.DeviceArray],
        action: jnp.DeviceArray,
    ):
        """ Rollout the next timestep with or without observations.

        If obs is available, the posterior distribution is used.
        If obs is none, the prior from the transition model is used instead.
        In both cases, the returned reconstruction is from the predicted latents for
        evaluation purpose.
        When observation is available, this corresponds to one-step prediction in the
        observation space.
        The latent prediction is made with either the frozen trained model or the
        adaptation model. The switch between the two is controlled by int_mask.

        Args:
            trained_params (dict): A dict containing the parameters of the pre-trained
                model.
            prev_step (dict): A dict containing infromation from the previous timestep.
            obs (Union[None, DeviceArray]): The observation at current timestep. Set to
                None if not available.
            action (DeviceArray): The action for the current timestep.

        Returns:
            carry (dict): A dict containing the relevant information to be carried to
                the next timestep.
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
        recon = self.trained_model.apply(
            trained_params, mean_prior, method=self.trained_model.decode
        )

        if obs is not None:
            # Use the posterior if observation is available.
            mean_q, _ = self.trained_model.apply(
                trained_params, obs, method=self.trained_model.encode
            )
            latent_error = (mean_prior - mean_q) ** 2
            z = mean_q
        else:
            # Use the prior if there is no observation.
            latent_error = None
            z = mean_prior

        # Predict the next timestep based on the estimated latent for the current
        # timestep.
        h_t, next_mu, next_logvar = self.trained_model.apply(
            trained_params,
            prev_step["hidden"],
            z,
            action,
            t_mask,
            method=self.trained_model.predict_next,
        )
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
            # Note that the masks are not re-sampled within a trajectory.
            "transition_mask": t_mask,
            "int_mask": int_mask,
            "rng": rng,
        }
        return carry, (recon, latent_error)

    @classmethod
    @jax.tree_util.Partial(jax.jit, static_argnums=(0, 6))
    def train(
        cls,
        trained_params: flax.core.FrozenDict,
        state: train_state.TrainState,
        data: Tuple[jnp.DeviceArray, jnp.DeviceArray],
        rng: jnp.DeviceArray,
        lambdas: dict,
        dimensions: Tuple[(int,) * 3] = (64, 24, 2),
    ) -> Tuple[Tuple[(jnp.DeviceArray,) * 5], train_state.TrainState]:
        def loss_fn(params):
            init_carry = cls.get_init_carry(
                trained_params, *dimensions, data[0][0], params["params"], rng
            )
            _, outputs = jax.lax.scan(
                lambda c, x: state.apply_fn(params, trained_params, c, x[0], x[1]),
                init_carry,
                data,
            )
            recon = outputs[0]
            recon_mse = ((recon - data[0]) ** 2).mean(axis=(0, 1, 2)).sum()
            kl = outputs[1].mean(axis=(0, 1, 2)).sum()
            l_error = outputs[2].mean(axis=1)
            int_loss = cls.intervention_sparsity(params)
            return (
                recon_mse + lambdas["kl"] * kl + lambdas["int"] * int_loss,
                (recon_mse, kl, l_error, 0, int_loss,),
            )

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return loss[1], state

    @classmethod
    @jax.tree_util.Partial(jax.jit, static_argnums=(0, 6))
    def evaluate(
        cls,
        trained_params: flax.core.FrozenDict,
        state: train_state.TrainState,
        data: Tuple[jnp.DeviceArray, jnp.DeviceArray],
        rng: jnp.DeviceArray,
        lambdas: dict,
        dimensions: Tuple[(int,) * 3] = (64, 24, 2),
    ) -> Tuple[Tuple[(jnp.DeviceArray,) * 5], train_state.TrainState]:
        def loss_fn(params):
            init_carry = cls.get_init_carry(
                trained_params, *dimensions, data[0][0], params["params"], rng
            )
            _, outputs = jax.lax.scan(
                lambda c, x: state.apply_fn(params, trained_params, c, x[0], x[1]),
                init_carry,
                data,
            )
            recon = outputs[0]
            recon_mse = ((recon - data[0]) ** 2).mean(axis=(0, 1, 2)).sum()
            kl = outputs[1].mean(axis=(0, 1, 2)).sum()
            l_error = outputs[2].mean(axis=1)
            int_loss = cls.intervention_sparsity(params)
            return (
                recon_mse + lambdas["kl"] * kl + lambdas["int"] * int_loss,
                (recon_mse, kl, l_error, 0, int_loss,),
            )

        return loss_fn(state.params)[1]

    def init_train_state(
        self,
        trained_params: flax.core.FrozenDict,
        rng: jnp.DeviceArray,
        batch: tuple[jnp.DeviceArray, jnp.DeviceArray],
        lr: float = 0.005,
    ):
        batched_obs = batch[0][0]
        mask = {"intervention_targets": jnp.ones((1, self.latent_dim))}
        init_carry = self.get_init_carry(
            trained_params,
            self.hidden_dim,
            self.latent_dim,
            self.action_dim,
            batched_obs,
            mask,
            rng,
        )
        params = self.init(rng, trained_params, init_carry, batch[0][0], batch[1][0])
        tx = optax.adam(learning_rate=lr)
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=tx)


class AdaptRSSM(BaseAdapt):
    trained_model: RSSM

    def setup(self):
        self.int_prior_net = nn.vmap(
            TransitionRNN,
            in_axes=(1, 1, None),
            out_axes=1,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim)
        self.int_prior = lambda h, z, a, mask: self.int_prior_net(
            h, jnp.concatenate([z, a], axis=-1), 1
        )

    @classmethod
    def get_init_carry(
        cls, trained_params, hidden_dim, latent_dim, action_dim, batch, params, rng
    ):
        # Initialise the hidden state and initial action to zeros.
        h_0 = jnp.zeros((batch.shape[0], batch.shape[1], hidden_dim))
        a_0 = jnp.zeros((batch.shape[0], batch.shape[1], action_dim))

        # The initial prior distribution is unit normal.
        mu, logvar = (
            jnp.zeros((batch.shape[0], batch.shape[1], latent_dim)),
            jnp.zeros((batch.shape[0], batch.shape[1], latent_dim)),
        )
        rng, key = random.split(rng)

        # In adaptation for RSSM, the int_... model is used for latent prediction.
        # The model is initialised from the model parameters of the trained transition
        # model.
        carry = {
            "hidden": h_0,
            "prior_mu": mu,
            "prior_logvar": logvar,
            "int_hidden": h_0,
            "int_prior_mu": mu,
            "int_prior_logvar": logvar,
            "action": a_0,
            "transition_mask": 0,
            "int_mask": 1,  # set to 1 in order to use the new model.
            "rng": rng,
        }
        return carry

    def init_train_state(self, trained_params, rng, batch, lr):
        batched_obs = batch[0][0]
        mask = {"intervention_mask": jnp.ones((1, self.latent_dim))}
        init_carry = self.get_init_carry(
            trained_params,
            self.hidden_dim,
            self.latent_dim,
            self.action_dim,
            batched_obs,
            mask,
            rng,
        )
        params = self.init(rng, trained_params, init_carry, batch[0][0], batch[1][0])
        # for RSSM adaptation, initialise transition model weights with the trained
        # model.
        params = flax.core.unfreeze(params)
        params["params"]["int_prior_net"] = trained_params["params"]["prior_net"]
        params = flax.core.freeze(params)
        tx = optax.adam(learning_rate=lr)
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=tx)

    @staticmethod
    def sparsity(params):
        """ Returns 0 as there is no sparsity loss in RSSM."""
        return 0

    @staticmethod
    def intervention_sparsity(params):
        """ Returns 0 as there is no sparsity loss in RSSM."""
        return 0


class AdaptMultiRSSM(BaseAdapt):
    trained_model: MultiRSSM

    def setup(self):
        self.int_prior_net = nn.vmap(
            TransitionRNN,
            in_axes=(1, 1, None),
            out_axes=1,
            variable_axes={"params": 0},
            split_rngs={"params": True},
        )(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim)
        self.int_prior = lambda h, z, a, mask: self.int_prior_net(
            h, jnp.concatenate([z, a], axis=-1), 1
        )

    @classmethod
    def get_init_carry(
        cls, trained_params, hidden_dim, latent_dim, action_dim, batch, params, rng
    ):
        # batch is a batch of observations (batch_size, n_int, *obs_dim)
        h_0 = jnp.zeros((batch.shape[0], batch.shape[1], hidden_dim))
        a_0 = jnp.zeros((batch.shape[0], batch.shape[1], action_dim))
        mu, logvar = (
            jnp.zeros((batch.shape[0], batch.shape[1], latent_dim)),
            jnp.zeros((batch.shape[0], batch.shape[1], latent_dim)),
        )
        rng, key = random.split(rng)
        carry = {
            "hidden": h_0,
            "prior_mu": mu,
            "prior_logvar": logvar,
            "int_hidden": h_0,
            "int_prior_mu": mu,
            "int_prior_logvar": logvar,
            "action": a_0,
            "transition_mask": 0,
            "policy_mask": 0,
            "reward_mask": 0,
            "int_mask": 1,
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


class AdaptVCD(BaseAdapt):
    # the adapt vcd class needs reimplemented versions of __call__ and rollout funcitons
    # in order to use the trained VCD model.
    trained_model: VCD

    def setup(self):
        self.int_prior_net = nn.vmap(
            ParallelRNN,
            in_axes=(1, 1, 1),
            out_axes=1,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim)
        self.int_prior = lambda h, z, a, mask: self.int_prior_net(
            h, jnp.concatenate([z, a], axis=-1), mask
        )
        self.intervention_targets = self.param(
            "intervention_targets",
            lambda *x: 1 * jnp.ones((1, self.latent_dim)),
            (1, self.latent_dim),
        )

    def __call__(
        self,
        trained_params: flax.core.frozen_dict.FrozenDict,
        prev_step: dict,
        obs: jnp.DeviceArray,
        action: jnp.DeviceArray,
    ) -> tuple[dict, Tuple[(jnp.DeviceArray,) * 3]]:
        # expand the inputs in order to perform VCD predictions.
        obs = jnp.repeat(obs, self.n_env, axis=1)
        action = jnp.repeat(action, self.n_env, axis=1)
        rng = prev_step["rng"]
        t_mask = prev_step["transition_mask"]
        int_mask = prev_step["int_mask"]
        int_mean_p = prev_step["int_prior_mu"]
        int_logvar_p = prev_step["int_prior_logvar"]
        mean_p = prev_step["prior_mu"]
        logvar_p = prev_step["prior_logvar"]

        # encode and decode the observation using the trained model
        mean_q, logvar_q = self.trained_model.apply(
            trained_params, obs, method=self.trained_model.encode
        )
        mean_q = mean_q[:, [-1], :]
        logvar_q = logvar_q[:, [-1], :]
        z = self.trained_model.reparameterize(rng, mean_q, logvar_q)
        recon = self.trained_model.apply(
            trained_params, z, method=self.trained_model.decode
        )

        # Compute the KL div between prior (predicted by the adaptation model) and
        # posterior
        obs_kl = KL_div(mean_q, logvar_q, mean_p, logvar_p)
        int_kl = KL_div(mean_q, logvar_q, int_mean_p, int_logvar_p)

        # Compute the overall KL term under the full intervention model (eq. 24)
        kl = int_mask * int_kl + (1 - int_mask) * obs_kl
        latent_error = (
            int_mask * (int_mean_p - mean_q) ** 2
            + (1 - int_mask) * (mean_p - mean_q) ** 2
        )

        # Predict forward using the adaptation model for environment specific mechanisms
        # The trained model is used otherwise.
        h_t, next_mu, next_logvar = self.trained_model.apply(
            trained_params,
            prev_step["hidden"],
            z,
            action[:, [-1], :],
            t_mask,
            method=self.trained_model.predict_next,
        )
        int_h_t, int_next_mu, int_next_logvar = self.int_prior(
            prev_step["int_hidden"], z, action[:, [-1], :], t_mask
        )

        # update the random key to ensure different reparameterisation samples for next
        # timestep
        rng, key = random.split(rng)
        carry = {
            "hidden": h_t,
            "prior_mu": next_mu,
            "prior_logvar": next_logvar,
            "int_hidden": int_h_t,
            "int_prior_mu": int_next_mu,
            "int_prior_logvar": int_next_logvar,
            "action": action[:, [-1], :],
            "transition_mask": t_mask,
            "int_mask": int_mask,
            "rng": rng,
        }
        return carry, (recon, kl, latent_error)

    def rollout(
        self,
        trained_params: flax.core.FrozenDict,
        prev_step: dict,
        obs: Union[None, jnp.DeviceArray],
        action: jnp.DeviceArray,
    ):
        if obs is not None:
            obs = jnp.repeat(obs, self.n_env, axis=1)
        action = jnp.repeat(action, self.n_env, axis=1)
        rng = prev_step["rng"]
        t_mask = prev_step["transition_mask"]
        int_mask = prev_step["int_mask"]
        int_mean_p = prev_step["int_prior_mu"]
        mean_p = prev_step["prior_mu"]

        # Reconstruct using prior distribution from the transition model.
        mean_prior = int_mask * int_mean_p + (1 - int_mask) * mean_p
        recon = self.trained_model.apply(
            trained_params, mean_prior, method=self.trained_model.decode
        )

        if obs is not None:
            # Use the posterior if observation is available.
            mean_q, _ = self.trained_model.apply(
                trained_params, obs, method=self.trained_model.encode
            )
            mean_q = mean_q[:, [-1], :]
            latent_error = (mean_prior - mean_q) ** 2
            z = mean_q
        else:
            # Use the prior if there is no observation.
            latent_error = None
            z = mean_prior

        # Predict the next timestep based on the estimated latent for the current
        # timestep.
        h_t, next_mu, next_logvar = self.trained_model.apply(
            trained_params,
            prev_step["hidden"],
            z,
            action[:, [-1], :],
            t_mask,
            method=self.trained_model.predict_next,
        )
        int_h_t, int_next_mu, int_next_logvar = self.int_prior(
            prev_step["int_hidden"], z, action[:, [-1], :], t_mask
        )

        rng, key = random.split(rng)
        carry = {
            "hidden": h_t,
            "prior_mu": next_mu,
            "prior_logvar": next_logvar,
            "int_hidden": int_h_t,
            "int_prior_mu": int_next_mu,
            "int_prior_logvar": int_next_logvar,
            "action": action[:, [-1], :],
            # Note that the masks are not re-sampled within a trajectory.
            "transition_mask": t_mask,
            "int_mask": int_mask,
            "rng": rng,
        }
        return carry, (recon, latent_error)

    @classmethod
    def get_init_carry(
        cls, trained_params, hidden_dim, latent_dim, action_dim, batch, params, rng
    ):
        # batch is a batch of observations (batch_size, n_int, *obs_dim)
        h_0 = jnp.zeros((batch.shape[0], batch.shape[1], hidden_dim, latent_dim))
        a_0 = jnp.zeros((batch.shape[0], batch.shape[1], action_dim))

        rng1, rng2 = random.split(rng)
        t_mask = VCD.gumbel_max_state(
            trained_params["params"]["causal_graph"], batch, rng1
        )
        int_mask = VCD.gumbel_max_intervention(
            params["intervention_targets"], batch, rng2
        )

        mu, logvar = (
            jnp.zeros((batch.shape[0], batch.shape[1], latent_dim)),
            jnp.zeros((batch.shape[0], batch.shape[1], latent_dim)),
        )
        rng, key = random.split(rng1)
        carry = {
            "hidden": h_0,
            "prior_mu": mu,
            "prior_logvar": logvar,
            "int_hidden": h_0,
            "int_prior_mu": mu,
            "int_prior_logvar": logvar,
            "action": a_0,
            "transition_mask": t_mask,
            "int_mask": int_mask,
            "rng": rng,
        }
        return carry

    @staticmethod
    def intervention_sparsity(params):
        """Returns the expected number of intervention targets"""
        return jnp.sum(jax.nn.sigmoid(params["params"]["intervention_targets"]))
