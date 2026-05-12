"""
server.py
---------
Server-side FedAvg aggregation logic.

TASK 1: Clean FedAvg baseline.
TASK 2: Pass client_id and malicious info to client_update.
TASK 3: Pass label_flip_map dict instead of single source/target.

FUTURE: Krum, Median, and other defenses added here as alternative
        aggregation functions alongside fedavg_aggregate().
"""

import copy
import torch


def fedavg_aggregate(global_model, local_weights_list, client_sizes):
    """
    Aggregate local weights using FedAvg weighted average.
    w_global = Σ (n_k / n_total) * w_k
    UNCHANGED from Task 1.
    """
    total_samples = sum(client_sizes)

    avg_weights = copy.deepcopy(local_weights_list[0])
    for key in avg_weights:
        avg_weights[key] = torch.zeros_like(avg_weights[key], dtype=torch.float32)

    for weights, size in zip(local_weights_list, client_sizes):
        weight_factor = size / total_samples
        for key in avg_weights:
            avg_weights[key] += weight_factor * weights[key].float()

    global_model.load_state_dict(avg_weights)
    return global_model


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
    label_flip_map=None,      # ← single dict replaces source/target params
    verbose=True
):
    """
    Execute one complete FedAvg communication round.

    Args:
        round_num         : Current round number
        global_model      : Shared global model
        client_loaders    : List of DataLoaders, one per client
        client_update_fn  : client_update function from client.py
        local_epochs      : E — local epochs per round
        lr                : SGD learning rate
        device            : 'cpu' or 'cuda'
        client_fraction   : Fraction of clients selected per round
        malicious_clients : Set of malicious client IDs (None = clean)
        label_flip_map    : Dict {source: target} e.g. {7:1, 9:4, 3:8}
                            None = no attack (clean baseline)
        verbose           : Print per-client logs

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

    # ── Step 2: Local training ────────────────────────────────────
    local_weights_list = []
    client_sizes       = []
    round_losses       = []

    for client_id in selected_ids:
        loader = client_loaders[client_id]

        local_weights, avg_loss = client_update_fn(
            global_model      = global_model,
            client_loader     = loader,
            local_epochs      = local_epochs,
            lr                = lr,
            device            = device,
            client_id         = client_id,
            malicious_clients = malicious_clients,
            label_flip_map    = label_flip_map,   # ← updated
        )

        local_weights_list.append(local_weights)
        client_sizes.append(len(loader.dataset))
        round_losses.append(avg_loss)

        if verbose:
            role = ("MALICIOUS"
                    if (malicious_clients and client_id in malicious_clients)
                    else "honest")
            print(f"  Client {client_id:2d} [{role:9s}] | "
                  f"samples={len(loader.dataset):4d} | "
                  f"loss={avg_loss:.4f}")

    # ── Step 3: FedAvg aggregation ────────────────────────────────
    global_model   = fedavg_aggregate(global_model, local_weights_list, client_sizes)
    avg_round_loss = sum(round_losses) / len(round_losses)

    if verbose:
        print(f"[Round {round_num}] Avg client loss: {avg_round_loss:.4f}")

    return global_model, round_losses