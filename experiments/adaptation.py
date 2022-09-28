import os
import json
from flax import serialization
from jax import random
from jax import numpy as jnp
import tqdm


def adapt(
    train_data,
    test_data,
    adapt_model_class,
    model_class,
    model_path,
    checkpoint_id,
    mixing_matrix,
    epochs=5000,
    observation_length=100
):
    model_conf = json.load(open(os.path.join(model_path, "model_conf.json")))
    model = model_class(
        latent_dim=model_conf["latent_dim"],
        action_dim=2,
        hidden_dim=model_conf["hidden_dim"],
        obs_dim=12,
        n_env=6,
    )
    rng, key = random.split(random.PRNGKey(model_conf["random_seed"]))
    # load the trained model
    state = model.init_train_state(rng, train_data[0], lr=model_conf["lr"])
    state_dict = jnp.load(
        os.path.join(model_path, f"model_checkpoint_{checkpoint_id}.npy"),
        allow_pickle=True,
    ).item()["state_dict"]
    state = serialization.from_state_dict(state, state_dict)

    adapt_model = adapt_model_class(
        latent_dim=model_conf["latent_dim"],
        action_dim=2,
        hidden_dim=model_conf["hidden_dim"],
        obs_dim=12,
        n_env=model.n_env,
        trained_model=model,
    )

    lambdas = {
        "kl": 1,
        "sparse": 0,
        "int": 0.5,
    }
    dimensions = (model_conf["hidden_dim"], model_conf["latent_dim"], 2)

    print("training on adaptation data")
    tbar = tqdm.tqdm(range(epochs))
    adapt_state = adapt_model.init_train_state(
        state.params, rng, train_data[0], lr=0.005
    )
    for ep in tbar:
        batch = train_data[0]
        rng, _ = random.split(rng)
        loss, adapt_state = adapt_model.train(
            state.params, adapt_state, batch, rng, lambdas, dimensions
        )
        int_sparsity = adapt_model.intervention_sparsity(adapt_state.params)
        tbar.set_description(f"kl: {loss[1]:.2f}, int: {int_sparsity:.2f}")

    carry = adapt_model.get_init_carry(
        state.params,
        model_conf["hidden_dim"],
        model_conf["latent_dim"],
        2,
        test_data[0][0][0],
        adapt_state.params["params"],
        rng,
    )

    rollout_error = []
    episode_length = test_data[0][0].shape[0]
    print("evaluating")
    for i in tqdm.tqdm(range(episode_length)):
        error = []
        batch = test_data[0]
        if i <= observation_length:
            carry, out = adapt_model.apply(
                adapt_state.params,
                state.params,
                carry,
                batch[0][i],
                batch[1][i],
                method=adapt_model.rollout,
            )
        else:
            carry, out = adapt_model.apply(
                adapt_state.params,
                state.params,
                carry,
                None,
                batch[1][i],
                method=adapt_model.rollout,
            )
        if mixing_matrix is not None:
            error.append(
                (((out[0] - batch[0][i]) @ jnp.linalg.pinv(mixing_matrix)) ** 2)
                .sum(-1)
                .mean(-1)
            )
        else:
            error.append((((out[0] - batch[0][i]) ** 2).sum((2, 3, 4))))
        rollout_error.append(jnp.concatenate(error))

    return jnp.stack(rollout_error), adapt_state
