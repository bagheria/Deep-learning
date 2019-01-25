# https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

import numpy as np


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, 2 * (self.y - self.output) * sigmoid(self.output, True))
        d_weights1 = np.dot(self.input.T, np.dot(2 * (self.y - self.output) * sigmoid(self.output, True),
                                                 self.weights2.T) * sigmoid(self.layer1, True))

        self.weights1 += np.dot(d_weights1, 0.1)
        self.weights2 += np.dot(d_weights2, 0.1)

    def train(self):
        self.feedforward()
        self.backprop()


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
NN = NeuralNetwork(X, y)
for i in range(15000):  # trains the NN 1,000 times
    if i % 100 == 0:
        print("for iteration # " + str(i) + "\n")
        print("Input : \n" + str(X))
        print("Actual Output: \n" + str(y))
        print("Predicted Output: \n" + str(NN.output))
        print("Loss: \n" + str(np.mean(np.square(y - NN.output))))  # mean sum squared loss
        print("\n")

    NN.train()
