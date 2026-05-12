"""
utils.py
--------
Plotting, logging, and checkpoint utilities.

Keeps notebooks clean — all visualization logic lives here.
"""

import os
import json
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ─────────────────────────────────────────────────────────────────
# 1. Plot MTA across communication rounds
# ─────────────────────────────────────────────────────────────────

def plot_accuracy(
    accuracies,
    title="FedAvg Baseline — MNIST Main Task Accuracy",
    save_path=None
):
    """
    Plot accuracy vs. communication round.

    Args:
        accuracies : List of (float) accuracy values, one per round
        title      : Plot title
        save_path  : If provided, save the figure to this path (.png)
    """
    rounds = list(range(1, len(accuracies) + 1))

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(rounds, [a * 100 for a in accuracies],
            color='steelblue', linewidth=2, marker='o', markersize=4,
            label='Global MTA')

    # Horizontal reference lines
    ax.axhline(y=97, color='orange', linestyle='--', linewidth=1.2,
               label='97% threshold')
    ax.axhline(y=99, color='green', linestyle='--', linewidth=1.2,
               label='99% threshold')

    ax.set_xlabel("Communication Round", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, len(accuracies) + 1)
    ax.set_ylim(85, 101)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f%%'))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[Utils] Figure saved to: {save_path}")

    plt.show()


# ─────────────────────────────────────────────────────────────────
# 2. Plot per-class accuracy bar chart
# ─────────────────────────────────────────────────────────────────

def plot_per_class_accuracy(per_class_acc, title="Per-Class Accuracy", save_path=None):
    """
    Bar chart of accuracy for each digit class 0–9.

    Args:
        per_class_acc : Dict {class_id: accuracy_float}
        title         : Plot title
        save_path     : Optional save path
    """
    classes = list(per_class_acc.keys())
    values  = [per_class_acc[c] * 100 for c in classes]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(classes, values, color='steelblue', edgecolor='navy', alpha=0.8)

    # Label each bar with its value
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}%",
                ha='center', va='bottom', fontsize=9)

    ax.set_xlabel("Digit Class", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(classes)
    ax.set_ylim(0, 107)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)

    plt.show()


# ─────────────────────────────────────────────────────────────────
# 3. Save & Load Results
# ─────────────────────────────────────────────────────────────────

def save_results(results_dict, path="./results/baseline_results.json"):
    """
    Save experiment results to a JSON file for later comparison.

    Args:
        results_dict : Dictionary of results (e.g. {'mta': [...], 'config': {...}})
        path         : File path to save to
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"[Utils] Results saved to: {path}")


def load_results(path):
    """
    Load previously saved results from JSON.

    Args:
        path : Path to the JSON file

    Returns:
        results_dict : Dictionary of results
    """
    with open(path, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────
# 4. Save & Load Model Checkpoint
# ─────────────────────────────────────────────────────────────────

def save_model(model, path="./results/global_model.pt"):
    """
    Save global model weights to disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[Utils] Model saved to: {path}")


def load_model(model, path):
    """
    Load weights into an existing model instance.
    """
    model.load_state_dict(torch.load(path, map_location='cpu'))
    print(f"[Utils] Model loaded from: {path}")
    return model


# ─────────────────────────────────────────────────────────────────
# 5. Print a nicely formatted round summary
# ─────────────────────────────────────────────────────────────────

def print_round_summary(round_num, mta, loss, asr=None, best_mta=None):
    """
    Print a clean one-line summary after each round.
    Shows MTA always. Shows ASR only when attack is active.
    """
    star = " ★" if (best_mta is not None and mta >= best_mta) else "  "

    if asr is not None:
        print(f"  Round {round_num:3d}{star} | "
              f"MTA: {mta*100:6.2f}% | "
              f"ASR: {asr*100:6.2f}% | "
              f"Loss: {loss:.4f}")
    else:
        print(f"  Round {round_num:3d}{star} | "
              f"MTA: {mta*100:6.2f}% | "
              f"Loss: {loss:.4f}")
    

# ─────────────────────────────────────────────────────────────────
# Plot ASR across communication rounds  — Task 3
# ─────────────────────────────────────────────────────────────────

def plot_asr(
    asr_list,
    label_flip_map=None,
    per_mapping_history=None,
    title="Label-Flipping Attack — Attack Success Rate (ASR)",
    save_path=None
):
    """
    Plot ASR across communication rounds.
    Shows overall ASR always.
    If per_mapping_history is provided, also plots each mapping separately.

    Args:
        asr_list             : List of overall ASR floats, one per round
        label_flip_map       : Dict {src: tgt} — used for legend labels
        per_mapping_history  : List of per_mapping dicts (one per round)
                               Each entry: {(src,tgt): {'asr':...}}
        title                : Plot title
        save_path            : Optional save path (.png)
    """
    rounds = list(range(1, len(asr_list) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))

    # ── Overall ASR ───────────────────────────────────────────────
    ax.plot(rounds, [a * 100 for a in asr_list],
            color='crimson', linewidth=2.5, marker='o', markersize=5,
            label='Overall ASR', zorder=5)

    # ── Per-mapping ASR (if history is provided) ──────────────────
    mapping_colors = ['#E07B54', '#9B59B6', '#2ECC71', '#F39C12']

    if per_mapping_history and label_flip_map:
        for i, (src, tgt) in enumerate(label_flip_map.items()):
            key   = (src, tgt)
            color = mapping_colors[i % len(mapping_colors)]
            # Extract this mapping's ASR for each round
            mapping_asr_series = [
                round_data[key]['asr'] * 100
                for round_data in per_mapping_history
                if key in round_data
            ]
            if mapping_asr_series:
                ax.plot(rounds[:len(mapping_asr_series)],
                        mapping_asr_series,
                        color=color, linewidth=1.5,
                        marker='s', markersize=3.5,
                        linestyle='--',
                        label=f'ASR {src}→{tgt}',
                        alpha=0.8)

    # Reference line at 50%
    ax.axhline(y=50, color='gray', linestyle=':', linewidth=1.2,
               label='50% reference')

    ax.set_xlabel("Communication Round", fontsize=13)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, len(asr_list) + 1)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f%%'))
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[Utils] Figure saved to: {save_path}")

    plt.show()


# ─────────────────────────────────────────────────────────────────
# Combined MTA + ASR plot  — Task 3
# ─────────────────────────────────────────────────────────────────

def plot_mta_and_asr(
    mta_list,
    asr_list,
    title="FedAvg with Label-Flipping Attack",
    save_path=None
):
    """
    Plot both MTA and ASR on the same figure with two y-axes.

    This is the key visualization for the project:
        - MTA stays high  → global model still works overall
        - ASR rises       → attack is successfully degrading digit 7

    Args:
        mta_list  : List of MTA floats per round
        asr_list  : List of ASR floats per round
        title     : Plot title
        save_path : Optional path to save (.png)
    """
    assert len(mta_list) == len(asr_list), \
        "mta_list and asr_list must have the same length"

    rounds = list(range(1, len(mta_list) + 1))

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # ── Left axis: MTA (blue) ──────────────────────────────────
    color_mta = 'steelblue'
    ax1.set_xlabel("Communication Round", fontsize=13)
    ax1.set_ylabel("Main Task Accuracy (%)", fontsize=13, color=color_mta)
    ax1.plot(rounds, [m * 100 for m in mta_list],
             color=color_mta, linewidth=2, marker='o', markersize=4,
             label='MTA')
    ax1.tick_params(axis='y', labelcolor=color_mta)
    ax1.set_ylim(80, 102)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f%%'))

    # ── Right axis: ASR (red) ──────────────────────────────────
    ax2 = ax1.twinx()   # share the x-axis, new y-axis on the right
    color_asr = 'crimson'
    ax2.set_ylabel("Attack Success Rate (%)", fontsize=13, color=color_asr)
    ax2.plot(rounds, [a * 100 for a in asr_list],
             color=color_asr, linewidth=2, marker='s', markersize=4,
             linestyle='--', label='ASR (7→1)')
    ax2.tick_params(axis='y', labelcolor=color_asr)
    ax2.set_ylim(0, 105)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f%%'))

    # ── Combined legend ────────────────────────────────────────
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=11, loc='center right')

    ax1.set_title(title, fontsize=14)
    ax1.grid(True, alpha=0.25)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[Utils] Figure saved to: {save_path}")

    plt.show()