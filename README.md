# Federated Learning for Healthcare: HAR Dataset with FedProx Algorithm

This project implements federated learning for healthcare applications, focusing on the Human Activity Recognition (HAR) dataset. We employ the **FedProx** algorithm to enhance model training in a federated learning environment, specifically addressing challenges like non-IID data distribution and system heterogeneity. The goal is to deploy privacy-preserving machine learning models in healthcare scenarios while maintaining high accuracy.

## Introduction

Federated learning is a distributed machine learning technique that allows training models on decentralized data sources without the need to transfer raw data. In this project, we apply federated learning to healthcare data, specifically the **Human Activity Recognition (HAR)** dataset, using the **FedProx** algorithm to improve model convergence under non-IID (non-Independent and Identically Distributed) data.

The primary objective is to train a model on distributed data while ensuring privacy and security of healthcare data. We focus on enabling healthcare devices (e.g., wearables) to collaboratively learn without compromising sensitive information.

## Federated Learning

Federated learning involves training machine learning models across multiple devices or servers that hold local data, which eliminates the need to share raw data. Instead, model updates (gradients) are shared and aggregated in a central server. This process ensures privacy preservation and reduces data transfer costs.

### Key Features of Federated Learning:
- **Data Privacy:** Data is kept local to each device, ensuring user privacy.
- **Scalability:** Models can be trained across thousands of devices.
- **Efficiency:** Reduces data transfer by sharing only model parameters.

## FedProx Algorithm

FedProx is an advanced federated learning algorithm that addresses issues such as:
- **Client Heterogeneity:** Different devices or clients may have different computational resources or local data distributions.
- **Non-IID Data:** Data distributions across clients may differ significantly, making training challenging.

FedProx modifies the FedAvg (Federated Averaging) algorithm by introducing a proximal term to the objective function. This helps to stabilize training when clients have highly varied data or computational power.

### Benefits of FedProx:
- Improved convergence under non-IID data settings.
- Robustness against system heterogeneity.

## Dataset

We use the **Human Activity Recognition (HAR)** dataset, which consists of sensor data from smartphones used to record activities such as walking, sitting, and standing.

- **Data source:** [HAR Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- **Features:**
  - 561 features corresponding to time and frequency domain variables.
  - Target: 6 activity classes.

### Preprocessing:
- Data is normalized and split into training and testing sets.
- Each client in the federated learning setup receives a fraction of the data.

