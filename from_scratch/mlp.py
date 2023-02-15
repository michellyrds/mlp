from typing import Any, Dict

import numpy as np
from activation_functions import bipolar, d_sigmoid, sigmoid
from sklearn.model_selection import train_test_split


class MultilayerPerceptron(object):
    def __init__(self, hyperparameters: Dict[str, Any], seed=None):
        self.n_inputs = hyperparameters["n_inputs"]
        self.hidden_layers = hyperparameters["n_hidden_layers"]
        self.n_outputs = hyperparameters["n_outputs"]

        layers = [self.n_inputs] + self.hidden_layers + [self.n_outputs]

        np.random.seed(seed)

        weights = []
        for i in range(len(layers) - 1):
            w = np.random.uniform(-1, 1, (layers[i], layers[i + 1]))
            # w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)

        self.activations = activations

        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)

        self.derivatives = derivatives

    def split_dataset(self, dataset):
        X, y = dataset[:, : -self.n_outputs], dataset[:, self.n_inputs :]

        return X, y

    def forward_propagate(self, inputs):
        activations = inputs

        self.activations[0] = activations

        for i, w in enumerate(self.weights[:-1]):
            net_inputs = np.dot(activations, w)
            activations = sigmoid(net_inputs)
            self.activations[i + 1] = activations

        w = self.weights[-1]
        net_inputs = np.dot(activations, w)
        activations = bipolar(0, net_inputs)
        self.activations[-1] = activations
        # !
        return activations

    def back_propagate(self, error):
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i + 1]

            delta = error * d_sigmoid(activations)

            delta_t = delta.reshape(delta.shape[0], -1).T

            current_activations = self.activations[i]

            current_activations = current_activations.reshape(
                current_activations.shape[0], -1
            )

            self.derivatives[i] = np.dot(current_activations, delta_t)

            error = np.dot(delta, self.weights[i].T)

    def mean_square_error(self, label, output):
        return np.average((label - output) ** 2)

    def gradient_descent(self, learning_rate: float):
        for i in range(len(self.weights)):
            w = self.weights[i]
            derivatives = self.derivatives[i]
            w += derivatives * learning_rate
            self.weights[i] = w

    def train(
        self,
        dataset,
        maxEpochs,
        learning_rate,
        test_size,
        random_state=None,
        accMin=0.80,
    ):
        X, y = self.split_dataset(dataset)

        sum_error = 0

        error_rate_test = 1
        error_rate_test_ant = 1

        for i in range(maxEpochs):
            print(f"\n---------------- Epoch {i+1} ----------------")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            sum_error_train = 0

            for j, input in enumerate(X_train):
                output = self.forward_propagate(input)

                error = y_train[j] - output

                self.back_propagate(error)

                self.gradient_descent(learning_rate)

                sum_error_train += self.mean_square_error(y_train[j], output)

            error_rate_train = sum_error_train / (len(X_train))
            print(f"Training error: {error_rate_train}")

            sum_error_test = 0
            for j, input in enumerate(X_test):
                output = self.forward_propagate(input)
                error = y_test[j] - output

                sum_error_test += self.mean_square_error(y_test[j], output)

            error_rate_test_ant = error_rate_test
            error_rate_test = sum_error_test / (len(X_test))
            print(f"Test error: {error_rate_test}")

            sum_error += error_rate_test
            acc = 1 - (sum_error / (i + 1))

            # TODO: Fix early stopping condition.
            if (
                error_rate_test_ant < error_rate_test
                and abs((error_rate_test - error_rate_train)) < 0.15
                and acc >= accMin
            ):
                print(
                    f"[Early stopping] Training finished at epoch {i+1}"
                    f" with accuracy of {acc}"
                )
                return

        print(f"\nTraining finished. Model accuracy: {1-(sum_error/maxEpochs)}")

    def predict(self, inputs):
        output = []

        for input in inputs:
            output.append(self.forward_propagate(input))

        return np.asarray(output)
