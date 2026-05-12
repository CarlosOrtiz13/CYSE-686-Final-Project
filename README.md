# Federated Learning Label-Flipping Attack on MNIST

## Overview

This project implements a Federated Learning (FL) environment using the FedAvg algorithm on the MNIST dataset. It evaluates how multi-source label-flipping attacks affect a global MNIST classifier and compares attack performance against a defense using robust aggregation.

The project includes three experiments:

1. Baseline Label-Flipping Attack
2. Enhanced Label-Flipping Attack
3. Defense Against Label-Flipping Attack

The main goal is to measure whether malicious clients can increase Attack Success Rate (ASR) while keeping Main Task Accuracy (MTA) high.

---

# Project Structure

```text
├── notebooks/
│   ├── Baseline_flipping_attack.ipynb
│   ├── Enhanced_flipping_attack.ipynb
│   └── Defense_against_flipping_attack.ipynb
│
├── src/
│   ├── client.py
│   ├── data.py
│   ├── evaluation.py
│   ├── model.py
│   ├── server.py
│   └── utils.py
│
├── results/
│
└── README.md
```

---

# Requirements

Install the required Python libraries before running the notebooks.

```bash
pip install torch torchvision matplotlib numpy
```

Recommended Python version:

```text
Python 3.10+
```

GPU is optional. The code supports CUDA if a CUDA-enabled PyTorch installation is available.

---

# Important Note Before Running

Before running each notebook, restart the Jupyter kernel.

In Jupyter Notebook or VS Code, use:

```text
Kernel → Restart Kernel and Run All Cells
```

This is important because old variables, trained models, cached weights, or previous experiment results can remain in memory and affect the next run.

Each experiment should be run from a clean restarted kernel.

---

# Dataset

This project uses the MNIST handwritten digit dataset.

The dataset is downloaded automatically when running the notebooks.

---

# Model Architecture

The global model is a Convolutional Neural Network (CNN) for MNIST classification.

The model includes:

- 2 convolutional layers
- ReLU activation
- Max pooling
- 2 fully connected layers
- 10 output classes for digits 0–9

The model is implemented in:

```text
src/model.py
```

---

# Federated Learning Setup

The project uses the following general FL configuration:

| Parameter | Value |
|---|---|
| Number of clients | 20 |
| Client fraction per round | 1.0 |
| Communication rounds | 20 |
| Honest local epochs | 1 |
| Learning rate | 0.01 |
| Batch size | 32 |
| Test batch size | 512 |
| Data split | IID |
| Dataset | MNIST |

Each round follows this process:

1. The server sends the global model to selected clients.
2. Clients train locally on their private data.
3. Malicious clients perform label flipping.
4. Clients send updated weights back to the server.
5. The server aggregates client updates.
6. The global model is evaluated using MTA and ASR.

---

# Attack Configuration

The project uses a multi-source label-flipping attack.

## Label-flip mappings

```text
7 → 1
9 → 4
3 → 8
```

This means malicious clients train the model to classify:

- Digit 7 as digit 1
- Digit 9 as digit 4
- Digit 3 as digit 8

---

# Experiment 1: Baseline Label-Flipping Attack

Notebook:

```text
notebooks/Baseline_flipping_attack.ipynb
```

## Parameters

| Parameter | Value |
|---|---|
| Aggregation method | FedAvg |
| Malicious ratio | 0.2 |
| Number of malicious clients | 4 out of 20 |
| Malicious clients | [0, 3, 7, 8] |
| Honest local epochs | 1 |
| Malicious local epochs | 1 |
| Label mappings | 7→1, 9→4, 3→8 |

## Purpose

The baseline attack shows the effect of label flipping when malicious clients train with the same number of epochs as honest clients.

Expected behavior:

- MTA remains high
- ASR is relatively low compared to the enhanced attack

---

# Experiment 2: Enhanced Label-Flipping Attack

Notebook:

```text
notebooks/Enhanced_flipping_attack.ipynb
```

## Parameters

| Parameter | Value |
|---|---|
| Aggregation method | FedAvg |
| Malicious ratio | 0.4 |
| Number of malicious clients | 8 out of 20 |
| Malicious clients | [0, 1, 2, 3, 7, 8, 11, 16] |
| Honest local epochs | 1 |
| Malicious local epochs | 3 |
| Label mappings | 7→1, 9→4, 3→8 |

## Purpose

The enhanced attack increases the influence of malicious clients by using more malicious participants and more malicious local training epochs.

Before running, verify that the enhanced configuration is set to:

```python
MALICIOUS_LOCAL_EPOCHS = 3
MALICIOUS_RATIO = 0.4
AGGREGATION_METHOD = "fedavg"
```

If the malicious epoch value is configured inside `src/server.py`, verify that:

```python
malicious_local_epochs = 3
```

## Expected behavior

The enhanced attack should increase ASR compared to the baseline attack while attempting to keep MTA relatively high.

---

# Experiment 3: Defense Against Label-Flipping Attack

Notebook:

```text
notebooks/Defense_against_flipping_attack.ipynb
```

## Parameters

| Parameter | Value |
|---|---|
| Aggregation method | Krum |
| Malicious ratio | 0.4 |
| Number of malicious clients | 8 out of 20 |
| Malicious clients | [0, 1, 2, 3, 7, 8, 11, 16] |
| Honest local epochs | 1 |
| Malicious local epochs | 3 |
| Label mappings | 7→1, 9→4, 3→8 |

## Purpose

The defense experiment uses Krum as a robust aggregation method instead of FedAvg.

The goal is to reduce the influence of malicious updates and lower ASR while maintaining acceptable MTA.

Before running, verify that the defense configuration is set to:

```python
AGGREGATION_METHOD = "krum"
MALICIOUS_LOCAL_EPOCHS = 3
MALICIOUS_RATIO = 0.4
```

---

# Evaluation Metrics

## Main Task Accuracy (MTA)

MTA measures the normal classification accuracy of the global model on the clean MNIST test set.

```text
MTA = Correct Predictions / Total Test Samples
```

A high MTA means the model still performs well on the original classification task.

## Attack Success Rate (ASR)

ASR measures how often the model predicts the attacker’s target label for the selected source classes.

Example:

```text
7 → 1
```

For this mapping, ASR measures how often test images of digit 7 are classified as digit 1.

---

# Output Files

The notebooks save plots and result files inside the `results/` directory.

Expected outputs include:

- MTA plots
- ASR plots
- JSON results files
- Model checkpoints

---

# How to Reproduce All Experiments

Run the notebooks in the following order:

```text
1. notebooks/Baseline_flipping_attack.ipynb
2. notebooks/Enhanced_flipping_attack.ipynb
3. notebooks/Defense_against_flipping_attack.ipynb
```

Before running each notebook:

1. Restart the notebook kernel
2. Run all cells from top to bottom
3. Verify that the printed configuration matches the expected parameters

---

# Expected Comparison

| Experiment | Aggregation | Malicious Ratio | Malicious Epochs | Expected Result |
|---|---|---|---|---|
| Baseline Attack | FedAvg | 0.2 | 1 | High MTA, low ASR |
| Enhanced Attack | FedAvg | 0.4 | 3 | Higher ASR while maintaining high MTA |
| Defense | Krum | 0.4 | 3 | Lower ASR compared to enhanced attack |

---

# Authors

Carlos Ortiz Collao  
Nicolas Alexandr Dmitriev  

George Mason University  
CYSE 686 Final Project

---

# Notes

- This project is intended for educational and research purposes only.
- The implementation focuses on Federated Learning security, adversarial machine learning, label-flipping attacks, and robust aggregation defenses.
- CPU execution is supported.
- GPU execution is optional if CUDA is available.