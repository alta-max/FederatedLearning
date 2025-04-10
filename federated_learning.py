#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Federated Learning Script
This script will implement federated learning using FedProx algorithm.
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Ensure eager execution is enabled
tf.config.run_functions_eagerly(True)

# Load data
X_train = np.loadtxt('X_train.txt')
y_train = np.loadtxt('y_train.txt')
X_test = np.loadtxt('X_test.txt')
y_test = np.loadtxt('y_test.txt')

# Normalize data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape to (samples, timesteps, features)
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Make labels zero-indexed if needed
y_train_adjusted = y_train - 1
y_test_adjusted = y_test - 1


# Federated Learning Setup with FedProx
# Define parameters
input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])
num_classes = len(np.unique(y_train_adjusted))
num_clients = 5
num_rounds = 10
local_epochs = 3
mu = 0.01  # Proximal term coefficient (hyperparameter for FedProx)

# Function to create LSTM model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Create the global model
global_model = build_model()

# Split data among clients
client_data = []
samples_per_client = len(X_train_reshaped) // num_clients

for i in range(num_clients):
    start = i * samples_per_client
    end = (i + 1) * samples_per_client if i < num_clients - 1 else len(X_train_reshaped)
    client_data.append((
        X_train_reshaped[start:end],
        y_train_adjusted[start:end]
    ))

# Alternative approach for FedProx: Custom training loop with proximal term
# Federated Learning Process with FedProx
# Track metrics
history = {
    'global_accuracy': [],
    'client_accuracies': [[] for _ in range(num_clients)],
    'proximal_terms': [[] for _ in range(num_clients)]
}

# Run federated learning for multiple rounds
for round_num in range(num_rounds):
    print(f"\n--- Round {round_num+1}/{num_rounds} ---")

    # Get the global model's weights
    global_weights = global_model.get_weights()

    # List to hold updated weights from each client
    all_client_weights = []
    client_samples = []

    # Train on each client's data using FedProx
    for i, (client_x, client_y) in enumerate(client_data):
        print(f"Training on client {i+1}/{num_clients}")

        # Create a new model with global weights
        client_model = build_model()
        client_model.set_weights(global_weights)

        # Get initial client weights
        initial_weights = client_model.get_weights()

        # Create dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((client_x, client_y))
        train_dataset = train_dataset.batch(32)

        # Create validation dataset (20% of client data)
        val_size = int(0.2 * len(client_x))
        val_x = client_x[-val_size:]
        val_y = client_y[-val_size:]
        client_x = client_x[:-val_size]
        client_y = client_y[:-val_size]

        # Create optimizer
        optimizer = tf.keras.optimizers.Adam()

        # Custom training loop with proximal term
        proximal_values = []
        val_accuracies = []

        for epoch in range(local_epochs):
            print(f"Epoch {epoch+1}/{local_epochs}")

            batch_losses = []
            proximal_terms = []

            # Training loop
            for batch_x, batch_y in train_dataset:
                with tf.GradientTape() as tape:
                    # Forward pass
                    logits = client_model(batch_x, training=True)

                    # Calculate loss
                    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
                    original_loss = loss_fn(batch_y, logits)

                    # Calculate proximal term
                    proximal_term = 0
                    for w1, w2 in zip(client_model.get_weights(), initial_weights):
                        proximal_term += tf.reduce_sum(tf.square(w1 - w2))

                    # Total loss
                    total_loss = original_loss + (mu / 2) * proximal_term

                # Get gradients and update weights
                grads = tape.gradient(total_loss, client_model.trainable_weights)
                optimizer.apply_gradients(zip(grads, client_model.trainable_weights))

                batch_losses.append(float(total_loss))
                proximal_terms.append(float(proximal_term))

            # Evaluate on validation data
            val_logits = client_model.predict(val_x)
            val_preds = np.argmax(val_logits, axis=1)
            val_acc = np.sum(val_preds == val_y) / len(val_y)
            val_accuracies.append(val_acc)

            avg_loss = np.mean(batch_losses)
            avg_proximal = np.mean(proximal_terms)
            proximal_values.append(avg_proximal)

            print(f"Loss: {avg_loss:.4f}, Proximal Term: {avg_proximal:.4f}, Val Accuracy: {val_acc:.4f}")

        # Store client's updated weights and sample count
        all_client_weights.append(client_model.get_weights())
        client_samples.append(len(client_x))

        # Record client validation accuracy (last epoch)
        history['client_accuracies'][i].append(val_accuracies[-1])
        history['proximal_terms'][i].append(proximal_values[-1])

        # Clean up to free memory
        tf.keras.backend.clear_session()

    # Federated Averaging (same as before)
    # Calculate the total number of samples
    total_samples = sum(client_samples)

    # Initialize new global weights
    avg_weights = [np.zeros_like(weight) for weight in global_weights]

    # Compute weighted average of client weights
    for i, client_weight in enumerate(all_client_weights):
        # Weight by the proportion of samples this client has
        weight_ratio = client_samples[i] / total_samples

        # Add the weighted client weights to our running average
        for j in range(len(avg_weights)):
            avg_weights[j] += client_weight[j] * weight_ratio

    # Update the global model with the new averaged weights
    global_model.set_weights(avg_weights)

    # Evaluate global model on test data
    test_loss, test_acc = global_model.evaluate(X_test_reshaped, y_test_adjusted, verbose=0)
    history['global_accuracy'].append(test_acc)

    print(f"Round {round_num+1} complete. Global model test accuracy: {test_acc:.4f}")

# Visualization and Evaluation
# Plot the global model accuracy over rounds
plt.figure(figsize=(12, 6))

# Plot global accuracy
plt.subplot(1, 2, 1)
plt.plot(range(1, num_rounds+1), history['global_accuracy'], marker='o')
plt.title('FedProx Global Model Accuracy')
plt.xlabel('Round')
plt.ylabel('Test Accuracy')
plt.grid(True)

# Plot client accuracies
plt.subplot(1, 2, 2)
for i in range(num_clients):
    plt.plot(range(1, num_rounds+1), history['client_accuracies'][i], marker='x', label=f'Client {i+1}')
plt.title('FedProx Client Validation Accuracies')
plt.xlabel('Round')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('fedprox_learning_results.png')
plt.show()

# Plot proximal terms
plt.figure(figsize=(10, 6))
for i in range(num_clients):
    plt.plot(range(1, num_rounds+1), history['proximal_terms'][i], marker='o', label=f'Client {i+1}')
plt.title('FedProx Proximal Term Values')
plt.xlabel('Round')
plt.ylabel('Proximal Term Value')
plt.legend()
plt.grid(True)
plt.savefig('fedprox_proximal_terms.png')
plt.show()

# Get predictions from final global model
y_pred = np.argmax(global_model.predict(X_test_reshaped), axis=1)

# Create confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

cm = confusion_matrix(y_test_adjusted, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('FedProx Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('fedprox_confusion_matrix.png')
plt.show()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test_adjusted, y_pred))

# Save the final global model
global_model.save('fedprox_har_model.h5')
print("\nTraining complete! Final model saved as 'fedprox_har_model.h5'")
