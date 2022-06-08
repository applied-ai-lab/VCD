from sys import path

path.append("..")
import json
import os
from datetime import datetime
import argparse
import jax.random as random


def train_step(
    state, model, rng, train_data, val_data, lambdas, dimensions, writer, iter_idx
):
    training_recon, training_kl, training_latent_error = 0, 0, 0
    for batch in train_data:
        rng, val_rng = random.split(rng)
        loss, state = model.train(state, batch, rng, lambdas, dimensions)
        training_recon += loss[0].item()
        training_kl += loss[1].item()
        training_latent_error += loss[2].mean(0).sum().item()
    writer.add_scalar("recon/training", training_recon / len(train_data), iter_idx)
    writer.add_scalar("kl/training", training_kl / len(train_data), iter_idx)
    writer.add_scalar(
        "error/training", training_latent_error / len(train_data), iter_idx
    )
    writer.add_scalar("sparsity", loss[3].item(), iter_idx)
    writer.add_scalar("intervention_targets", loss[4].item(), iter_idx)
    val_recon, val_kl, val_latent_error = 0, 0, 0
    for batch in val_data:
        rng, val_rng = random.split(val_rng)
        loss = model.evaluate(state, batch, val_rng, lambdas, dimensions)
        val_recon += loss[0].item()
        val_kl += loss[1].item()
        val_latent_error += loss[2].mean(0).sum().item()
    writer.add_scalar("recon/val", val_recon / len(val_data), iter_idx)
    writer.add_scalar("kl/val", val_kl / len(val_data), iter_idx)
    writer.add_scalar("error/val", val_latent_error / len(val_data), iter_idx)
    return state, rng
