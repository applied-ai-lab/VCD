from sys import path

path.append("..")
import json
import os
from datetime import datetime
import argparse
from tqdm import tqdm
import shutil
import tensorboardX
from flax import serialization
from data.generate_data import get_states
from data.dataset import DataLoader
from models.RSSM import RSSM
from models.MultiRSSM import MultiRSSM
from models.VCD import VCD
from training import train_step
import jax
import jax.random as random
import jax.numpy as jnp

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_conf", type=str, default="../data/mixed_state_data_conf.json"
)
parser.add_argument(
    "--model_conf", type=str, default="../models/mixed_state_RSSM_conf.json"
)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--checkpoint_freq", type=int, default=100)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

data_conf = json.load(open(args.data_conf))
model_conf = json.load(open(args.model_conf))

# ---------- setting up datasets ----------
if args.verbose:
    print("Generating dataset.")
mixing_matrix = random.normal(random.PRNGKey(0), (8, 12))
mixing_function = lambda x: x @ mixing_matrix
train_data_config = data_conf["train_data_conf"]
train_data = DataLoader(
    get_states, train_data_config, 100, data_conf["train_data_seed"], mixing_function
)
val_data_config = data_conf["val_data_conf"]
val_data = DataLoader(
    get_states, val_data_config, 100, data_conf["val_data_seed"], mixing_function
)

# ---------- setting up model ----------
if model_conf["model"] == "RSSM":
    m_class = RSSM
elif model_conf["model"] == "MultiRSSM":
    m_class = MultiRSSM
elif model_conf["model"] == "VCD":
    m_class = VCD
else:
    raise NotImplementedError

dimensions = (model_conf["hidden_dim"], model_conf["latent_dim"], 2)
lambdas = {
    "kl": 1,
    "sparse": model_conf["lambda_sparse"],
    "int": model_conf["lambda_intervention"],
}
model = m_class(
    latent_dim=model_conf["latent_dim"],
    action_dim=2,
    hidden_dim=model_conf["hidden_dim"],
    obs_dim=12,
    n_env=len(data_conf["train_data_conf"]["interventions"]),
)
rng, key = random.split(random.PRNGKey(model_conf["random_seed"]))
state = model.init_train_state(rng, train_data[0], lr=model_conf["lr"])
n_epochs = args.epochs

# ---------- setting up logger ----------
if args.run_name is None:
    run_name = (
        datetime.now().strftime("%h_%d__%H_%M_%S") + model.__class__.__name__ + "/"
    )
else:
    run_name = args.run_name
try:
    log_dir = os.path.join(os.environ["LOG_DIR"], run_name)
except:
    log_dir = os.path.join("../runs/", run_name)

writer = tensorboardX.SummaryWriter(log_dir)
shutil.copyfile(args.model_conf, os.path.join(log_dir, "model_conf.json"))
shutil.copyfile(args.data_conf, os.path.join(log_dir, "data_conf.json"))
iter_idx = 0

# ---------- Training Loop ----------
for epoch in tqdm(range(n_epochs), disable=not (args.verbose)):
    state, rng = train_step(
        state, model, rng, train_data, val_data, lambdas, dimensions, writer, iter_idx
    )
    iter_idx += 1
    if iter_idx % args.checkpoint_freq == 0:
        if isinstance(model, VCD):
            writer.add_image(
                "causal_graph",
                jnp.expand_dims(
                    jax.nn.sigmoid(state.params["params"]["causal_graph"]), 0
                ),
                iter_idx,
            )
            writer.add_image(
                "intervention_graph",
                jnp.expand_dims(
                    jax.nn.sigmoid(state.params["params"]["intervention_targets"]), 0
                ),
                iter_idx,
            )

        jnp.save(
            os.path.join(log_dir, f"model_checkpoint_{iter_idx}"),
            {"state_dict": serialization.to_state_dict(state), "iter_idx": iter_idx},
        )

