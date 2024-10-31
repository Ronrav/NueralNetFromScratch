import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def extract_data_from_csv(filename, label_index='last', scale=True):
    data = pd.read_csv(filename)
    data = data.values
    np.random.shuffle(data)
    # Split features and labels
    if label_index == 'last':
        X = data[:, :-1]
        y = data[:, -1]
    else:
        X = data[:, 1:]
        y = data[:, 0]
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    return X, y


def load_and_process_MNIST_data():

    X_train, y_train = extract_data_from_csv('../MNIST-train.csv')
    X_test, y_test = extract_data_from_csv('../MNIST-test.csv')

    X_train, X_test = X_train.astype(float), X_test.astype(float)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = one_hot_encode(y_train, num_classes=10)
    y_test = one_hot_encode(y_test, num_classes=10)

    return X_train, X_test, y_train, y_test


def one_hot_encode(labels, num_classes):
    """
    Convert an array of labels to one-hot encoded format.

    Parameters:
    labels (np.array): Array of integer labels.
    num_classes (int): The number of classes.

    Returns:
    np.array: One-hot encoded matrix.
    """
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


def transform_mb_labels(labels, pos="Pt_Fibro", neg="Pt_Ctrl"):
    transformed_labels = []
    for label in labels:
        if label.startswith(pos):
            transformed_labels.append(0)
        elif label.startswith(neg):
            transformed_labels.append(1)
        else:
            raise ValueError(f"Unexpected label format: {label}")
    return np.array(transformed_labels)

