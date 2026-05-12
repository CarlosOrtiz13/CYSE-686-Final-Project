Federated Learning Label-Flipping Attack on MNIST
Overview

This project implements a Federated Learning (FL) environment using the FedAvg algorithm on the MNIST dataset and evaluates the impact of multi-source label-flipping attacks performed by malicious clients.

The project contains two attack implementations:

Baseline Label-Flipping Attack
Enhanced Label-Flipping Attack

The objective of the project is to analyze how malicious federated clients can poison the global model while maintaining high overall classification accuracy.

Project Structure
├── notebooks/
│   ├── baseline_flip_attack.ipynb
│   └── enhanced_flip_attack.ipynb
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
Requirements

Install the required Python libraries before running the project.

pip install torch torchvision matplotlib numpy

Recommended Python version:

Python 3.10+
Dataset

This project uses the MNIST handwritten digit dataset.

The dataset is automatically downloaded when running the notebooks.

Model Architecture

The global model is a Convolutional Neural Network (CNN) composed of:

2 Convolutional layers
ReLU activations
MaxPooling layers
2 Fully Connected layers

The model implementation can be found in:

src/model.py
Federated Learning Setup

The project uses the FedAvg aggregation algorithm.

Workflow
The server initializes the global model.
MNIST is split across federated clients.
Selected clients perform local training.
Malicious clients perform label flipping.
Clients send updated local weights to the server.
The server aggregates updates using FedAvg.
The global model is evaluated using:
Main Task Accuracy (MTA)
Attack Success Rate (ASR)
Baseline Attack

The baseline attack performs multi-source label flipping using malicious clients.

Attack mappings
7 → 1
9 → 4
3 → 8
Malicious clients
[0, 3, 7, 8]
Run the baseline attack

Open and execute:

notebooks/baseline_flip_attack.ipynb
Enhanced Attack

The enhanced attack strengthens the influence of malicious clients by increasing the number of local training epochs used by malicious participants.

The enhanced attack uses the same attack mappings:

7 → 1
9 → 4
3 → 8

but malicious clients train more aggressively to increase the Attack Success Rate (ASR).

How to Run the Enhanced Attack

Open and execute:

notebooks/enhanced_flip_attack.ipynb

Before running the notebook, open:

src/server.py

and modify the following parameter:

malicious_local_epochs = 3

This enables the enhanced attack configuration by increasing the number of local epochs used by malicious clients.

Evaluation Metrics
Main Task Accuracy (MTA)

Measures the overall classification accuracy on the clean MNIST test set.

MTA = Correct Predictions / Total Test Samples
Attack Success Rate (ASR)

Measures how often the model predicts the attacker’s target label for poisoned source classes.

Files Description
File	Description
notebooks/baseline_flip_attack.ipynb	Runs the baseline attack experiment
notebooks/enhanced_flip_attack.ipynb	Runs the enhanced attack experiment
src/data.py	Downloads and splits MNIST across clients
src/model.py	CNN architecture
src/client.py	Local client training logic
src/server.py	FedAvg aggregation and communication rounds
src/evaluation.py	MTA and ASR evaluation
src/utils.py	Plotting and utility functions
Expected Results

The baseline attack maintains high Main Task Accuracy while producing a relatively low Attack Success Rate.

The enhanced attack attempts to increase ASR while maintaining a high MTA to remain stealthy.

Authors

Carlos Ortiz Collao
Nicolas Alexandr Dmitriev
George Mason University
CYSE 686 Final Project

Notes
This project is intended for educational and research purposes only.
The implementation focuses on Federated Learning security and adversarial machine learning concepts.
CPU execution is supported.
GPU execution is optional if CUDA is available.