"""
Credit to 'Machine Learning with PyTorch and Scikit-Learn'
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from neural_net import NeuralNetMLP, int_to_onehot

X, y = fetch_openml('mnist_784', version=1, parser='auto',
                    return_X_y=True)

X = X.values
y = y.astype(int).values

print(X.shape)
print(y.shape)

"""Normalize pixel values to [-1,1] rather than [0,255]"""
X = ((X / 225.) - .5) * 2

"""Show all digit categories"""
fig, ax = plt.subplots(nrows=2, ncols=5,
                       sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10):
    img = X[y == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

"""Show single example variants"""

fig, ax = plt.subplots(nrows=5,
                       ncols=5,
                       sharex=True,
                       sharey=True)
ax = ax.flatten()

for i in range(25):
    img = X[y == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

"""Divide into training(55,000 n), validation(5,000 n), and test(10,000 n) sets"""
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=10000, random_state=123, stratify=y)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=5000, random_state=123, stratify=y_temp)

"""Instantiate nn model"""
model = NeuralNetMLP(num_features=28 * 28,
                     num_hidden=0,
                     num_classes=10)

num_epochs = 50
minibatch_size = 100


def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        yield X[batch_idx], y[batch_idx]


"""Ensure mini-batches execute"""
for i in range(num_epochs):
    minibatch_gen = minibatch_generator(
        X_train, y_train, minibatch_size)
    for X_train_mini, y_train_mini, in minibatch_gen:
        break
    break

print(X_train_mini.shape)
print(y_train_mini.shape)

"""Define loss function and performance metric"""


def mse_loss(targets, probas, num_labels=10):
    onehot_targets = int_to_onehot(
        targets, num_labels=num_labels
    )
    return np.mean((onehot_targets - probas) ** 2)


def accuracy(targets, predicted_labels):
    return np.mean(predicted_labels == targets)


"""Test preceding functions by computing initial validation set mean squared error and accuracy"""
_, probas = model.forward(X_valid)
mse = mse_loss(y_valid, probas)

print(f'Initial Validation MSE: {mse:.1f}')

predicted_labels = np.argmax(probas, axis=1)
acc = accuracy(y_valid, predicted_labels)
print(f'Initial Validation Accuracy: {acc * 100:.1f}')
"""Expected accuracy of 10 or so given randomness with 10 classes"""

"""Compute MSE and accuracy incrementally as to avoid cpu over-usage"""


def compute_mse_and_acc(nnet, X, y, num_labels=10, minibatch_size=100):
    mse, correct_pred, num_examples = 0., 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)

    for i, (features, targets) in enumerate(minibatch_gen):
        _, probas = nnet.forward(features)
        predicted_labels = np.argmax(probas, axis=1)

        onehot_targets = int_to_onehot(targets, num_labels=num_labels)
        loss = np.mean((onehot_targets - probas) ** 2)
        correct_pred += (predicted_labels == targets).sum()

        num_examples += targets.shape[0]
        mse += loss

    mse = mse / i
    acc = correct_pred / num_examples
    return mse, acc


"""Test initial training set MSE and acc"""
mse, acc = compute_mse_and_acc(model, X_valid, y_valid)
print(f'Initial valid MSE: {mse:.1f}')
print(f'Initial valid accuracy: {acc * 100:.1f}%')

"""Train Model"""


def train(model, X_train, y_train, X_valid, y_valid, num_epochs, learning_rate=0.1):
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []

    for e in range(num_epochs):
        """iterate mini batches"""
        minibatch_gen = minibatch_generator(
            X_train, y_train, minibatch_size)
        for X_train_mini, y_train_mini in minibatch_gen:
            """Compute outputs"""
            a_h, a_out = model.forward(X_train_mini)

            """Compute Gradients"""

            d_loss__d_w_out, d_loss__d_b_out, \
            d_loss__d_w_h, d_loss__d_b_h = \
                model.backward(X_train_mini, a_h, a_out,
                               y_train_mini)

            """Update weights"""
            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
            model.weight_out -= learning_rate * d_loss__d_w_out
            model.bias_out -= learning_rate * d_loss__d_b_out


        """Log Epochs"""

        train_mse, train_acc = compute_mse_and_acc(
            model, X_train, y_train)

        valid_mse, valid_acc = compute_mse_and_acc(
            model, X_valid, y_valid)

        train_acc, valid_acc = train_acc*100, valid_acc*100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)
        print(f'Epoch: {e+1:03d}/{num_epochs:03d}'
              f'| Train MSE: {train_mse:.2f}'
              f'| Train Acc: {train_acc:.2f}%'
              f'| Valid Acc: {valid_acc:.2f}%')

    return epoch_loss, epoch_train_acc, epoch_valid_acc

np.random.seed(123)
epoch_loss, epoch_train_acc, epoch_valid_acc = train(model,
                                                     X_train,
                                                     y_train,
                                                     X_valid,
                                                     y_valid,
                                                     num_epochs=50,
                                                     learning_rate=0.1)








