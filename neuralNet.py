import numpy as np
from sklearn.model_selection import KFold
import activationFunctions as af
import matplotlib.pylab as plt
from sklearn.model_selection import LeaveOneOut

activation_functions = {
    'sigmoid': (af.sigmoid, af.sigmoid_d),
    'softmax': (af.softmax, None),  # Softmax derivative is handled differently
    'relu': (af.relu, af.relu_d),
    'tanh': (af.tanh, af.tanh_d)
}


class NeuralNetwork:
    def __init__(self, layers, act_funcs, loss_function='mse', lmbda=0.00):
        # Initialize the neural network with the given parameters
        self.layers = layers  # list containing the number of units in each layer
        self.act_funcs = act_funcs  # list of activation functions for each layer
        self.weights = [0] * (len(self.layers) - 1)  # initialize weights
        self.biases = [0] * (len(self.layers) - 1)  # initialize biases
        self._init_weights()  # call the method to initialize weights and biases
        self.costs = []  # list to store the cost during training
        self.lmbda = lmbda  # regularization parameter
        if loss_function not in ['mse', 'cross_entropy']:
            raise ValueError('Unsupported loss function:' + loss_function)
        self.loss_function = loss_function  # set the loss function
        self.n = 0  # number of training samples

    def _init_weights(self):
        # Initialize weights and biases
        for i in range(len(self.layers) - 1):
            self.weights[i] = np.random.randn(self.layers[i], self.layers[i + 1]) * 0.1  # random weights
            self.biases[i] = np.random.randn(1, self.layers[i + 1]) * 0.1  # random biases

    def mse_loss(self, y_true, y_pred):
        # Mean Squared Error loss function
        return np.mean(np.power(y_true - y_pred, 2))

    def cross_entropy_loss(self, y_true, y_pred):
        # Cross Entropy loss function
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-9)) / m

    def feedforward(self, x):
        # Perform the feedforward operation
        activations = [x]  # list to store activations of each layer
        zs = []  # list to store linear combinations (z) of each layer
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(x, w) + b  # compute linear combination
            zs.append(z)
            # get layer i activation function
            activation_func = activation_functions[self.act_funcs[i]][0]
            x = activation_func(z)  # apply activation function
            activations.append(x)
        return activations, zs

    def backprop(self, x, y):
        # Perform the backpropagation algorithm
        activations, zs = self.feedforward(x)  # get activations and zs from feedforward
        y_pred = activations[-1]  # prediction
        delta = (y_pred - y) / y.shape[0]  # compute initial delta
        deltas = [delta]  # list to store deltas
        if self.loss_function == 'mse':
            activation_derivative = activation_functions[self.act_funcs[-1]][1]
            deltas = [delta * activation_derivative(zs[-1])]  # adjust delta for mse loss

        for i in range(len(self.layers) - 2, 0, -1):
            activation_derivative = activation_functions[self.act_funcs[i-1]][1]
            delta = deltas[-1].dot(self.weights[i].T) * activation_derivative(zs[i-1])  # compute delta for previous layer
            deltas.append(delta)  # store delta

        deltas.reverse()  # reverse deltas to match layer order

        grad_w = []  # list to store gradients of weights
        grad_b = []  # list to store gradients of biases
        for i in range(len(self.weights)):
            grad_w.append(activations[i].T.dot(deltas[i]))  # compute gradient for weights
            grad_b.append(np.sum(deltas[i], axis=0, keepdims=True))  # compute gradient for biases

        return grad_w, grad_b  # return gradients

    def update_weights(self, grad_w, grad_b, learning_rate):
        # Update weights and biases using the gradients and learning rate
        for i in range(len(self.weights)):
            # support L2 regularization
            self.weights[i] = self.weights[i] * (1 - learning_rate * self.lmbda / self.n) - (learning_rate * grad_w[i])
            self.biases[i] -= learning_rate * grad_b[i]

    def train(self, X, y, epochs=1000, learning_rate=0.01, batch_size=30, verbose=False):
        # Train the neural network
        self.n = len(X)  # set number of training samples
        for epoch in range(epochs):
            permutation = np.random.permutation(X.shape[0])
            X = X[permutation]  # shuffle training data
            y = y[permutation]  # shuffle labels
            for start in range(0, X.shape[0], batch_size):
                end = min(start + batch_size, X.shape[0])
                X_batch = X[start:end]  # get batch of training data
                y_batch = y[start:end]  # get batch of labels
                grad_w, grad_b = self.backprop(X_batch, y_batch)  # compute gradients using backpropagation
                self.update_weights(grad_w, grad_b, learning_rate)  # update weights and biases

            if verbose:
                if epoch % 5 == 0:
                    y_pred = self.feedforward(X)[0][-1]  # get predictions
                    if self.loss_function == 'mse':
                        loss = self.mse_loss(y, y_pred)  # compute loss
                    elif self.loss_function == 'cross_entropy':
                        loss = self.cross_entropy_loss(y, y_pred)  # compute loss
                    self.costs.append(loss)  # store loss
                    print(f'Epoch {epoch}, Loss: {loss}')  # print loss

    def predict(self, x):
        # Make predictions
        return self.feedforward(x)[0][-1]

    def score(self, X, y):
        # Compute accuracy
        predictions = self.predict(X)  # get predictions
        y_hat = np.argmax(predictions, axis=1)  # get predicted labels
        labels = np.argmax(y, axis=1)  # get true labels
        accuracy = np.mean(y_hat == labels)  # compute accuracy
        return accuracy

    def k_fold_cross_validation(self, X, y, k=5, epochs=500, learning_rate=0.1, batch_size=10):
        # Perform k-fold cross-validation
        kf = KFold(n_splits=k)
        fold_scores = []  # list to store scores of each fold

        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]  # split data into training and validation sets
            y_train, y_val = y[train_index], y[val_index]  # split labels into training and validation sets
            self._init_weights()  # reinitialize weights and biases
            self.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)  # train on training set
            score = self.score(X_val, y_val)  # evaluate on validation set
            fold_scores.append(score)  # store validation score


        average_score = np.mean(fold_scores)  # compute average validation score
        print(f'Average Validation Score: {average_score}')
        return average_score

    def leave_one_out_cross_validation(self, X, y, epochs=500, learning_rate=0.1, batch_size=10):
        # Perform leave-one-out cross-validation
        loo = LeaveOneOut()
        fold_scores = []  # list to store scores of each fold
        fold_costs = []  # list to store costs of each fold

        for train_index, val_index in loo.split(X):
            X_train, X_val = X[train_index], X[val_index]  # split data into training and validation sets
            y_train, y_val = y[train_index], y[val_index]  # split labels into training and validation sets
            self._init_weights()  # reinitialize weights and biases
            self.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)  # train on training set
            score = self.score(X_val, y_val)  # evaluate on validation set
            fold_scores.append(score)  # store validation score

        average_score = np.mean(fold_scores)  # compute average validation score
        print(f'Average Validation Score: {average_score}')
        return average_score

    def plot_cost(self, hop=5):
        # Plot the cost during training
        plt.figure()
        x_values = [i * hop for i in range(len(self.costs))]  # x-axis values
        plt.plot(x_values, self.costs)  # plot costs
        plt.xlabel("epochs")  # label x-axis
        plt.ylabel("Loss")  # label y-axis
        plt.title(self.loss_function)  # set plot title
        plt.grid()  # add grid to plot
        plt.show()  # display plot



