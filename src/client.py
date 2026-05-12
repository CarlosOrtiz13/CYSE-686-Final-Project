"""
client.py
---------
Local training logic for a single federated client.

TASK 1: Clean FedAvg — honest clients only.
TASK 2: Label-flipping attack — malicious clients flip source_label → target_label
        BEFORE the loss computation, during local training only.
        The test set is NEVER modified.

FUTURE:  More sophisticated attacks (e.g. scaling, backdoor triggers) go here.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim


# ─────────────────────────────────────────────────────────────────
# Core client training function
# ─────────────────────────────────────────────────────────────────

def client_update(
    global_model,
    client_loader,
    local_epochs,
    lr,
    device,
    client_id=None,           # NEW: which client is this
    malicious_clients=None,   # NEW: set of malicious client IDs
    label_flip_map=None,      # dict mapping {source: target} e.g. {7:1, 9:4, 3:8}
):
    """
    Perform local training on a single client.

    For HONEST clients: standard SGD on local data, no modifications.
    For MALICIOUS clients: flip source_label → target_label in every
    batch BEFORE computing the loss. The global model and test set
    are never touched.

    Args:
        global_model      : Current global model (sent by server)
        client_loader     : DataLoader for this client's local data
        local_epochs      : Number of local training epochs (E)
        lr                : Learning rate for local SGD
        device            : 'cpu' or 'cuda'
        client_id         : Integer ID of this client
        malicious_clients : Set/list of malicious client IDs
        label_flip_map    : Dict mapping source_label → target_label for attack

    Returns:
        local_weights : OrderedDict of updated model parameters
        avg_loss      : Average training loss over the last epoch
    """

    # ── Determine if this client is malicious ─────────────────────
    # Default to empty set if not provided (safe for clean baseline)
    if malicious_clients is None:
        malicious_clients = set()

    is_malicious = (client_id in malicious_clients)

    # Only print the flip message ONCE per client (not every batch)
    if is_malicious:
        mappings_str = ", ".join(f"{s}→{t}" for s, t in label_flip_map.items())
        print(f"  [Attack] Client {client_id} is MALICIOUS — "
              f"flipping labels: {mappings_str}")

    # ── Step 1: Deep-copy global model so we don't modify it ──────
    local_model = copy.deepcopy(global_model)
    local_model.to(device)
    local_model.train()

    # ── Step 2: Set up optimizer and loss function ─────────────────
    optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Track how many labels were flipped this round (for logging)
    total_flipped = 0

    # ── Step 3: Run local epochs ───────────────────────────────────
    for epoch in range(local_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for images, labels in client_loader:
            images = images.to(device)
            labels = labels.to(device)

            # ── LABEL-FLIPPING ATTACK ──────────────────────────────
            # Only malicious clients reach this block.
            # We flip labels BEFORE computing the loss so the model
            # learns to misclassify source_label as target_label.
            # The test set is NEVER modified — only training labels.
            if is_malicious and label_flip_map:
                # Apply every mapping in the dict: 7→1, 9→4, 3→8
                # We flip a COPY of labels to avoid overwriting
                # a source label that is also someone else's target
                flipped_labels = labels.clone()
                for src, tgt in label_flip_map.items():
                    mask = (labels == src)
                    total_flipped += mask.sum().item()
                    flipped_labels[mask] = tgt
                labels = flipped_labels
            # ── END ATTACK BLOCK ───────────────────────────────────

            optimizer.zero_grad()
            outputs = local_model(images)       # forward pass
            loss = criterion(outputs, labels)   # loss on (possibly poisoned) labels
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

    avg_loss = epoch_loss / max(num_batches, 1)

    # ── Step 4: Return updated weights ────────────────────────────
    local_weights = local_model.state_dict()
    return local_weights, avg_loss


# ─────────────────────────────────────────────────────────────────
# Select a random subset of clients for each round
# ─────────────────────────────────────────────────────────────────

def select_clients(num_total_clients, fraction, seed=None):
    """
    Randomly select a fraction of clients to participate in a round.

    Args:
        num_total_clients : Total number of clients in the federation
        fraction          : Fraction to select (e.g. 1.0 = all clients)
        seed              : Optional random seed for reproducibility

    Returns:
        selected_ids : Sorted list of client indices
    """
    import random
    if seed is not None:
        random.seed(seed)

    num_selected = max(1, int(fraction * num_total_clients))
    selected_ids = random.sample(range(num_total_clients), num_selected)
    return sorted(selected_ids)


# ─────────────────────────────────────────────────────────────────
# NEW: Select malicious clients deterministically
# ─────────────────────────────────────────────────────────────────

def select_malicious_clients(num_total_clients, malicious_ratio, seed=42):
    """
    Select a fixed set of malicious clients based on a ratio.
    Uses a fixed seed so the same clients are malicious every round
    (realistic: attacker controls specific devices, not random ones).

    Args:
        num_total_clients : Total number of clients
        malicious_ratio   : Fraction to make malicious (e.g. 0.2 = 20%)
        seed              : Random seed for reproducibility

    Returns:
        malicious_set : Set of malicious client IDs
    """
    import random
    rng = random.Random(seed)   # isolated RNG — won't affect other randomness

    num_malicious = max(1, int(malicious_ratio * num_total_clients))
    malicious_list = rng.sample(range(num_total_clients), num_malicious)
    malicious_set = set(malicious_list)

    print(f"[Attack] {num_malicious} malicious clients "
          f"({malicious_ratio*100:.0f}% of {num_total_clients}): "
          f"{sorted(malicious_set)}")

    return malicious_set