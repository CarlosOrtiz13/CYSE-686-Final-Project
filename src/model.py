"""
model.py
--------
Defines the CNN architecture used for MNIST classification.

Architecture: Two conv layers + two fully-connected layers.
Chosen to be:
  - Expressive enough to reach ~99% on MNIST
  - Simple enough to train quickly on CPU
  - Easy to extend (e.g. add BatchNorm for later experiments)

TASK 1: Clean baseline — no modifications needed here for attacks/defenses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistCNN(nn.Module):
    """
    Simple CNN for MNIST digit classification.

    Input  : (batch, 1, 28, 28)  — grayscale MNIST images
    Output : (batch, 10)          — logits for 10 digit classes

    Layer breakdown:
        conv1  : 1  → 32 filters, 5×5 kernel  → (batch, 32, 24, 24)
        pool   : 2×2 max-pool                  → (batch, 32, 12, 12)
        conv2  : 32 → 64 filters, 5×5 kernel  → (batch, 64,  8,  8)
        pool   : 2×2 max-pool                  → (batch, 64,  4,  4)
        flatten: 64×4×4 = 1024
        fc1    : 1024 → 512
        fc2    : 512  → 10   (logits)
    """

    def __init__(self):
        super(MnistCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Block 1: Conv → ReLU → MaxPool
        x = F.relu(self.conv1(x))          # (B, 32, 24, 24)
        x = F.max_pool2d(x, kernel_size=2) # (B, 32, 12, 12)

        # Block 2: Conv → ReLU → MaxPool
        x = F.relu(self.conv2(x))          # (B, 64,  8,  8)
        x = F.max_pool2d(x, kernel_size=2) # (B, 64,  4,  4)

        # Flatten → Dense layers
        x = x.view(x.size(0), -1)          # (B, 1024)
        x = F.relu(self.fc1(x))            # (B, 512)
        x = self.fc2(x)                    # (B, 10)  ← raw logits

        return x


def get_model():
    """
    Factory function: returns a freshly initialized MnistCNN.
    Using a factory makes it easy to reset the global model between experiments.

    Returns:
        model : MnistCNN instance (randomly initialized weights)
    """
    return MnistCNN()


def count_parameters(model):
    """
    Count trainable parameters in a model (useful for sanity checks).

    Returns:
        int : Total number of trainable parameters
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {total:,}")
    return total


# ─────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = get_model()
    print(model)
    count_parameters(model)

    # Forward pass test
    dummy = torch.randn(4, 1, 28, 28)   # batch of 4 fake MNIST images
    out = model(dummy)
    print(f"[Check] Output shape: {out.shape}")  # expect (4, 10)