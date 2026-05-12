"""
server.py
---------
Server-side FedAvg aggregation logic.

TASK 1: Clean FedAvg baseline.
TASK 2: Pass client_id and malicious info to client_update.
TASK 3: Pass label_flip_map dict for multi-source attack.
TASK 4: Boosted Multi-Source Attack — malicious clients train longer
        locally (malicious_local_epochs > local_epochs) to amplify
        the effect of poisoned gradients before FedAvg aggregation.

FUTURE: Krum, Median, and other defenses added here alongside
        fedavg_aggregate() as alternative aggregation functions.
"""

import copy
import torch


# ─────────────────────────────────────────────────────────────────
# FedAvg Aggregation — UNCHANGED from Task 1
# ─────────────────────────────────────────────────────────────────

def fedavg_aggregate(global_model, local_weights_list, client_sizes):
    """
    Aggregate local weights using FedAvg weighted average.
    w_global = Σ (n_k / n_total) * w_k
    UNCHANGED — no defense modifications yet.
    """
    total_samples = sum(client_sizes)

    avg_weights = copy.deepcopy(local_weights_list[0])
    for key in avg_weights:
        avg_weights[key] = torch.zeros_like(
            avg_weights[key], dtype=torch.float32
        )

    for weights, size in zip(local_weights_list, client_sizes):
        weight_factor = size / total_samples
        for key in avg_weights:
            avg_weights[key] += weight_factor * weights[key].float()

    global_model.load_state_dict(avg_weights)
    return global_model


# ─────────────────────────────────────────────────────────────────
# One Full Communication Round — supports boosted malicious epochs
# ─────────────────────────────────────────────────────────────────

def run_round(
    round_num,
    global_model,
    client_loaders,
    client_update_fn,
    local_epochs,
    lr,
    device,
    client_fraction=1.0,
    malicious_clients=None,
    label_flip_map=None,
    malicious_local_epochs=3,   # ← NEW: if set, malicious clients
                                   #        train for this many epochs
                                   #        instead of local_epochs
    verbose=True
):
    """
    Execute one complete FedAvg communication round.

    Boosted attack logic:
        honest clients   → train for local_epochs
        malicious clients → train for malicious_local_epochs
                            (if malicious_local_epochs is set,
                             otherwise falls back to local_epochs)

    Args:
        round_num              : Current round number
        global_model           : Shared global model
        client_loaders         : List of DataLoaders, one per client
        client_update_fn       : client_update from client.py
        local_epochs           : E for honest clients
        lr                     : SGD learning rate
        device                 : 'cpu' or 'cuda'
        client_fraction        : Fraction of clients per round
        malicious_clients      : Set of malicious client IDs
        label_flip_map         : Dict {src: tgt} for label flipping
        malicious_local_epochs : E for malicious clients (boosting)
        verbose                : Print per-client logs

    Returns:
        global_model : Updated global model after aggregation
        round_losses : List of avg losses from each client
    """
    from src.client import select_clients

    num_clients = len(client_loaders)

    # ── Step 1: Select participating clients ──────────────────────
    selected_ids = select_clients(
        num_total_clients = num_clients,
        fraction          = client_fraction,
        seed              = round_num
    )

    if verbose:
        print(f"\n[Round {round_num}] {len(selected_ids)} clients selected")

    # ── Step 2: Local training on each selected client ────────────
    local_weights_list = []
    client_sizes       = []
    round_losses       = []

    for client_id in selected_ids:
        loader = client_loaders[client_id]

        # ── Per-client epoch selection (core of boosted attack) ───
        # Malicious clients train longer on poisoned data so their
        # gradient updates are more aggressively optimized toward
        # the attack objective before FedAvg averages them out.
        is_malicious_client = (
            malicious_clients is not None
            and client_id in malicious_clients
            and malicious_local_epochs is not None
        )
        client_epochs = (
            malicious_local_epochs if is_malicious_client
            else local_epochs
        )
        # ─────────────────────────────────────────────────────────

        local_weights, avg_loss = client_update_fn(
            global_model      = global_model,
            client_loader     = loader,
            local_epochs      = client_epochs,   # per-client epochs
            lr                = lr,
            device            = device,
            client_id         = client_id,
            malicious_clients = malicious_clients,
            label_flip_map    = label_flip_map,
        )

        local_weights_list.append(local_weights)
        client_sizes.append(len(loader.dataset))
        round_losses.append(avg_loss)

        if verbose:
            role = (
                "MALICIOUS"
                if (malicious_clients and client_id in malicious_clients)
                else "honest"
            )
            print(f"  Client {client_id:2d} [{role:9s}] | "
                  f"local_epochs={client_epochs} | "
                  f"samples={len(loader.dataset):4d} | "
                  f"loss={avg_loss:.4f}")

    # ── Step 3: FedAvg aggregation ────────────────────────────────
    global_model   = fedavg_aggregate(
        global_model, local_weights_list, client_sizes
    )
    avg_round_loss = sum(round_losses) / len(round_losses)

    if verbose:
        print(f"[Round {round_num}] Avg client loss: {avg_round_loss:.4f}")

    return global_model, round_losses