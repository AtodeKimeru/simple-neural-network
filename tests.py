from unittest import TestCase

from neural_network import NeuralNetwork, __sigmoid


class NeuralNetworkTest(TestCase):
    def test_initial_weights_in_range(self):
        """weights must be in range 0 to 1"""
        weight = NeuralNetwork().weights
        checks = 0
        for i in weight:
            if 0 < i < 1:
                checks += 1
        self.assertTrue(checks == 3)


    def test_sigmoid(self):
        __sigmoid()
