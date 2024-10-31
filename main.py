import utills
from neuralNet import NeuralNetwork


def classify_mnist():
    """
    Train Neural Network on MNIST train set, predict results on test set and prints the score
    """
    X_train, X_test, y_train, y_test = utills.load_and_process_MNIST_data()
    layers = [784, 256, 128, 10]
    activations = ['relu', 'relu','relu', 'sigmoid']
    nn = NeuralNetwork(layers=layers, act_funcs=activations, loss_function='mse', lmbda=5.0)
    nn.train(X_train, y_train, epochs=40, learning_rate=1.5, batch_size=64)

    score = nn.score(X_test, y_test)
    print(f'score on MNIST test set is: {score}')


if __name__ == '__main__':

    classify_mnist()





