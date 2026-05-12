"""
evaluation.py
-------------
Evaluation metrics for the federated learning experiments.

TASK 1: Main Task Accuracy (MTA) — measures how well the global
        model classifies MNIST digits on the held-out test set.

FUTURE:  Attack Success Rate (ASR) will be added here to measure
         how often the model misclassifies a backdoor trigger class.
"""

import torch


# ─────────────────────────────────────────────────────────────────
# Main Task Accuracy (MTA)
# ─────────────────────────────────────────────────────────────────

def evaluate_mta(model, test_loader, device):
    """
    Evaluate Main Task Accuracy (MTA) on the global test set.

    MTA = (# correctly classified test samples) / (# total test samples)

    This is the standard measure of whether the federated model is
    learning to classify digits correctly.

    Args:
        model       : The global model (MnistCNN)
        test_loader : DataLoader for the MNIST test set
        device      : 'cpu' or 'cuda'

    Returns:
        accuracy : Float in [0, 1] — fraction correctly classified
        loss     : Average cross-entropy loss on the test set
    """
    import torch.nn as nn

    model.eval()   # disable dropout / batchnorm training mode
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    num_batches = 0

    # No gradient computation needed during evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)                         # forward pass
            loss = criterion(outputs, labels)               # compute loss

            # Count correct predictions
            _, predicted = torch.max(outputs, dim=1)       # argmax over classes
            correct = (predicted == labels).sum().item()

            total_correct += correct
            total_samples += labels.size(0)
            total_loss += loss.item()
            num_batches += 1

    accuracy = total_correct / total_samples
    avg_loss = total_loss / max(num_batches, 1)

    return accuracy, avg_loss


# ─────────────────────────────────────────────────────────────────
# Per-class Accuracy Breakdown (optional diagnostic)
# ─────────────────────────────────────────────────────────────────

def evaluate_per_class(model, test_loader, device, num_classes=10):
    """
    Compute per-class accuracy on the test set.

    Useful for diagnosing whether the model has collapsed on any class,
    and later for verifying which class the label-flip attack targets.

    Args:
        model       : The global model
        test_loader : DataLoader for test set
        device      : 'cpu' or 'cuda'
        num_classes : Number of output classes (10 for MNIST)

    Returns:
        per_class_acc : Dict mapping class_id → accuracy (float)
    """
    model.eval()
    model.to(device)

    class_correct = [0] * num_classes
    class_total   = [0] * num_classes

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)

            for cls in range(num_classes):
                mask = (labels == cls)
                class_correct[cls] += (predicted[mask] == labels[mask]).sum().item()
                class_total[cls]   += mask.sum().item()

    per_class_acc = {}
    for cls in range(num_classes):
        if class_total[cls] > 0:
            per_class_acc[cls] = class_correct[cls] / class_total[cls]
        else:
            per_class_acc[cls] = 0.0

    return per_class_acc

# ─────────────────────────────────────────────────────────────────
# Attack Success Rate (ASR)  — Task 3
# ─────────────────────────────────────────────────────────────────

def evaluate_asr(model, test_loader, label_flip_map, device):
    """
    Evaluate Attack Success Rate (ASR) for a multi-source label-flip attack.

    For EACH mapping (source → target) in label_flip_map:
        mapping_asr = (# true-source samples predicted as target)
                      / (# total true-source samples in test set)

    Overall ASR = average of all per-mapping ASRs.

    Example with {7:1, 9:4, 3:8}:
        7→1 : 980 true-7s, 178 predicted as 1  → 18.2%
        9→4 : 1009 true-9s, 128 predicted as 4 → 12.7%
        3→8 : 1010 true-3s, 216 predicted as 8 → 21.4%
        Overall ASR = (18.2 + 12.7 + 21.4) / 3 = 17.4%

    IMPORTANT:
        - Uses the CLEAN test set (never poisoned)
        - Only examines samples whose true label == source
        - High per-mapping ASR = that specific flip is working
        - Overall ASR = average attack effectiveness across all flips

    Args:
        model          : The current global model
        test_loader    : DataLoader for the clean test set
        label_flip_map : Dict {source_label: target_label}
                         e.g. {7: 1, 9: 4, 3: 8}
        device         : 'cpu' or 'cuda'

    Returns:
        overall_asr    : Float in [0,1] — average ASR across all mappings
        per_mapping    : Dict {(src, tgt): {'asr', 'flipped', 'total'}}
                         detailed results for each individual mapping
    """
    model.eval()
    model.to(device)

    # ── Initialize counters for each mapping ──────────────────────
    # Structure: {(src, tgt): {'flipped': int, 'total': int}}
    counters = {
        (src, tgt): {'flipped': 0, 'total': 0}
        for src, tgt in label_flip_map.items()
    }

    # ── Single pass through the test set ──────────────────────────
    # One pass is enough — we check all mappings simultaneously
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs   = model(images)
            _, predicted = torch.max(outputs, dim=1)

            # For each mapping, check this batch
            for (src, tgt), cnt in counters.items():

                # Samples in this batch whose TRUE label is src
                source_mask = (labels == src)
                cnt['total'] += source_mask.sum().item()

                # Among those, how many were predicted as tgt?
                flipped_mask = source_mask & (predicted == tgt)
                cnt['flipped'] += flipped_mask.sum().item()

    # ── Compute per-mapping ASR ────────────────────────────────────
    per_mapping = {}
    asr_values  = []

    for (src, tgt), cnt in counters.items():
        if cnt['total'] == 0:
            mapping_asr = 0.0
            print(f"[Warning] No test samples with true label {src}.")
        else:
            mapping_asr = cnt['flipped'] / cnt['total']

        per_mapping[(src, tgt)] = {
            'asr'    : mapping_asr,
            'flipped': cnt['flipped'],
            'total'  : cnt['total'],
        }
        asr_values.append(mapping_asr)

    # ── Overall ASR = average across all mappings ─────────────────
    overall_asr = sum(asr_values) / len(asr_values) if asr_values else 0.0

    return overall_asr, per_mapping