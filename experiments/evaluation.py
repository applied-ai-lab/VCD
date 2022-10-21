import os
import json
from flax import serialization
from jax import random
from jax import numpy as jnp
import tqdm


def state_rollout_error(
    dataloader, model_class, run_path, checkpoint_id, observation_steps, mixing_matrix
):
    model_conf = json.load(open(os.path.join(run_path, "model_conf.json")))
    model = model_class(
        latent_dim=model_conf["latent_dim"],
        action_dim=2,
        hidden_dim=model_conf["hidden_dim"],
        obs_dim=12,
        n_env=dataloader[0][0].shape[2],
    )

    rng, key = random.split(random.PRNGKey(model_conf["random_seed"]))
    state = model.init_train_state(rng, dataloader[0], lr=model_conf["lr"])
    state_dict = jnp.load(
        os.path.join(run_path, f"model_checkpoint_{checkpoint_id}.npy"),
        allow_pickle=True,
    ).item()["state_dict"]
    state = serialization.from_state_dict(state, state_dict)

    rollout_error = []
    carry = model.get_init_carry(
        model_conf["hidden_dim"],
        model_conf["latent_dim"],
        2,
        dataloader[0][0][0],
        state_dict["params"]["params"],
        rng,
    )
    episode_length = dataloader[0][0].shape[0]
    for i in range(episode_length):
        error = []
        for batch in dataloader:
            if i <= observation_steps:
                carry, out = model.apply(
                    state.params, carry, batch[0][i], batch[1][i], method=model.rollout
                )
            else:
                carry, out = model.apply(
                    state.params, carry, None, batch[1][i], method=model.rollout
                )
            error.append(
                (((out[0] - batch[0][i]) @ jnp.linalg.pinv(mixing_matrix)) ** 2)
                .sum(-1)
                .mean(-1)
            )
        rollout_error.append(jnp.concatenate(error))
    return jnp.stack(rollout_error)


def image_rollout_error(
    dataloader, model_class, run_path, checkpoint_id, observation_steps, mixing_matrix
):
    model_conf = json.load(open(os.path.join(run_path, "model_conf.json")))
    model = model_class(
        latent_dim=model_conf["latent_dim"],
        action_dim=2,
        hidden_dim=model_conf["hidden_dim"],
        obs_dim=12,
        n_env=dataloader[0][0].shape[2],
    )

    rng, key = random.split(random.PRNGKey(model_conf["random_seed"]))
    state = model.init_train_state(rng, dataloader[0], lr=model_conf["lr"])
    state_dict = jnp.load(
        os.path.join(run_path, f"model_checkpoint_{checkpoint_id}.npy"),
        allow_pickle=True,
    ).item()["state_dict"]
    state = serialization.from_state_dict(state, state_dict)

    rollout_error = []
    carry = model.get_init_carry(
        model_conf["hidden_dim"],
        model_conf["latent_dim"],
        2,
        dataloader[0][0][0],
        state_dict["params"]["params"],
        rng,
    )
    episode_length = dataloader[0][0].shape[0]
    batch = dataloader[0]
    for i in tqdm.tqdm(range(episode_length)):
        error = []
        if i <= observation_steps:
            carry, out = model.apply(
                state.params, carry, batch[0][i], batch[1][i], method=model.rollout
            )
        else:
            carry, out = model.apply(
                state.params, carry, None, batch[1][i], method=model.rollout
            )
        error.append(
            (((out[0] - batch[0][i])**2)
            .sum((2,3,4))
        ))
        rollout_error.append(jnp.concatenate(error))
    return jnp.stack(rollout_error)