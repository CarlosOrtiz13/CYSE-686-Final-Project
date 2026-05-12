"""
data.py
-------
Handles downloading MNIST and splitting it across federated clients.

TASK 1: Clean FedAvg Baseline
- IID split only (each client gets a random, roughly equal share)
- Future tasks: add non-IID splits and label-flipping attacks here
"""

import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms


# ─────────────────────────────────────────────────────────────────
# 1. Download & Transform MNIST
# ─────────────────────────────────────────────────────────────────

def get_mnist(data_dir="./data"):
    """
    Download MNIST train and test sets with standard normalization.

    Returns:
        train_dataset : Full MNIST training set (60,000 samples)
        test_dataset  : Full MNIST test set (10,000 samples)
    """
    # Standard MNIST normalization values
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    print(f"[Data] Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    return train_dataset, test_dataset


# ─────────────────────────────────────────────────────────────────
# 2. IID Client Split
# ─────────────────────────────────────────────────────────────────

def iid_split(dataset, num_clients, seed=42):
    """
    Split a dataset evenly across clients in an IID fashion.
    Each client receives roughly equal samples with no class bias.

    Args:
        dataset     : PyTorch Dataset (e.g. MNIST train)
        num_clients : Number of federated clients
        seed        : Random seed for reproducibility

    Returns:
        client_datasets : List of Subset objects, one per client
    """
    total = len(dataset)
    samples_per_client = total // num_clients

    # Build split sizes; give any leftover samples to the last client
    sizes = [samples_per_client] * num_clients
    sizes[-1] += total - sum(sizes)

    client_datasets = random_split(
        dataset,
        sizes,
        generator=torch.Generator().manual_seed(seed)
    )

    print(f"[Data] IID split → {num_clients} clients, "
          f"~{samples_per_client} samples each")
    return client_datasets


# ─────────────────────────────────────────────────────────────────
# 3. Create DataLoaders
# ─────────────────────────────────────────────────────────────────

def get_client_loaders(client_datasets, batch_size=32):
    """
    Wrap each client dataset in a shuffled DataLoader for local training.

    Args:
        client_datasets : List of Subset datasets (from iid_split)
        batch_size      : Mini-batch size for local SGD

    Returns:
        client_loaders : List of DataLoader, one per client
    """
    client_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=True)
        for ds in client_datasets
    ]
    return client_loaders


def get_test_loader(test_dataset, batch_size=512):
    """
    Create a DataLoader for the global test set (MTA evaluation).

    Args:
        test_dataset : MNIST test set (10,000 samples)
        batch_size   : Batch size for inference (larger is faster)

    Returns:
        test_loader : DataLoader (not shuffled)
    """
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)