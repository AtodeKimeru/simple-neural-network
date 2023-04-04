from numpy import random, dot, exp

class NeuralNetwork:
    def __init__(self) -> None:
        random.seed(1)
        self.weights = 2 * random.random((3, 1)) - 1
        print(self.weights)


    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.weights))
    

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))


    def train(self, inputs, outputs, num ):
        for iteration in range(num):
            output = self.think(inputs)
            error = outputs - output
            adjustment = dot(inputs.T, error * output * (1-output))
            self.weights += adjustment
