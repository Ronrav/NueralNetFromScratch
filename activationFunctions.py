import numpy as np


def activate(x, name, deriv=False):
    if name == 'sigmoid':
        if deriv:
            return sigmoid_d(x)
        return sigmoid(x)
    elif name == 'tanh':
        if deriv:
            return tanh_d(x)
        return tanh(x)
    elif name == 'relu':
        if deriv:
            return relu_d(x)
        return relu(x)
    elif name == 'softmax':
        return softmax(x)
    else:
        raise ValueError("Unsupported activation function: " + name)


def sigmoid(x):
    """Sigmoid activation function."""
    x = np.clip(x, -500, 500)  # To avoid overflow in exp
    return 1 / (1 + np.exp(-x))


def sigmoid_d(x):
    """Derivative of the sigmoid activation function."""
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    """Tanh activation function."""
    return np.tanh(x)


def tanh_d(x):
    """Derivative of the tanh activation function."""
    return 1 - np.tanh(x)**2


def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)


def relu_d(x):
    """Derivative of the ReLU activation function."""
    return np.where(x > 0, 1, 0)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)





