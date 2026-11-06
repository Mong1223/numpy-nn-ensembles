import numpy as np

class my_NN(object):
    def __init__(self):
        self.input = 13  # feature numbers
        self.output = 1  # class number
        self.hidden_units = 13  # single layer

        # initialize matrix of weights;
        # np.random.seed(1)
        # weight1: input -> hidden layer
        self.w1 = np.random.randn(self.hidden_units, self.input) * 0.01  # 8*8 matrix
        self.b1 = np.zeros((self.hidden_units, 1))
        # weight2: hidden layer -> output
        self.w2 = np.random.randn(self.output, self.hidden_units) * 0.01  # 8*1 matrix
        self.b2 = np.zeros((self.output, 1))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_prime(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _forward_propagation(self, X):
        self.z2 = np.dot(self.w1, X) + self.b1
        self.a2 = self._sigmoid(self.z2)
        self.z3 = np.dot(self.w2, self.a2) + self.b2
        self.a3 = self._sigmoid(self.z3)
        return self.a3

    def _loss(self, predict, y):
        m = y.shape[1]
        epsilon = 1e-9
        logprobs = np.sum(np.multiply(y, np.log(predict)) + np.multiply((1 - y), np.log(1 - predict)))
        #
        # logprobs = np.average(np.multiply(y,np.log(predict + epsilon)) + np.multiply((1 - y), np.log(1 - predict + epsilon)))
        loss = - logprobs / m
        return loss

    def _backward_propagation(self, X, y):
        predict = self._forward_propagation(X)
        m = X.shape[1]
        delta3 = predict - y

        self.dw2 = (1 / m) * np.dot(delta3, self.a2.T)
        self.db2 = (1 / m) * np.sum(delta3, axis=1, keepdims=True)

        delta2 = np.multiply(np.dot(self.w2.T, delta3), self._sigmoid_prime(self.z2))
        self.dw1 = (1 / m) * np.dot(delta2, X.T)
        self.db1 = (1 / m) * np.sum(delta2, axis=1, keepdims=True)

    def _update(self, learning_rate=.1):
        self.w1 = self.w1 - learning_rate * self.dw1
        self.b1 = self.b1 - learning_rate * self.db1
        self.w2 = self.w2 - learning_rate * self.dw2
        self.b2 = self.b2 - learning_rate * self.db2

    def train(self, X, y, iteration=10000):
        for i in range(iteration):
            y_hat = self._forward_propagation(X)
            loss = self._loss(y_hat, y)
            self._backward_propagation(X, y)
            self._update()
            if i % 100 == 0:
                print("loss: ", loss)

    def predict(self, X):
        y_hat = self._forward_propagation(X)
        y_hat = [1. if i[0] >= 0.5 else 0. for i in y_hat.T]
        return np.array(y_hat)

    def score(self, predict, y):
        print(predict, ' : ', y[0])
        cnt = np.sum(predict == y[0])
        return (cnt / y.shape[1]) * 100
