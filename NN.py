import numpy as np
from .functions import *

class NeuralNetwork:
    """
    Layer-Customizable Neural Network

    This NN is implemented with vector-matrix modelling of Neurons.
    Dependent on Numpy.

    Attributes:
        learn_rate: the learning rate of the model.
        step: the learning step of the model.
        layers: the layer configuration of the model.
        matrices: the matrices used to model the neurons in the model.
    """

    def __init__(self, layers, learn_rate=0.01, step=1000):
        self.learn_rate=learn_rate
        self.step=step
        self.layers=layers
        
        # Function definition
        # Note that the parameter of derivative functions is the output of that function
        self.threshold_function = Tanh
        self.d_threshold = dTanh
        
        self.output_function = Linear
        self.d_output = dLinear
        
        self.loss_function = SquareError
        self.d_loss = dSquareError
        
        self.matrices = {}
        self.init_matrices()

    def init_matrices(self):
        """
        Initialize the matrices using Normal Distribution randomize
        """
        for i in range(1, len(self.layers)):
            self.matrices[f'Weight{i}'] = np.random.randn(self.layers[i-1], self.layers[i])
            self.matrices[f'Bias{i}'] = np.random.randn(self.layers[i],)

    def predict(self, input_matrix):
        """
        Predict the output given an input matrix
        A quick and least-productive forward propagation

        Args:
            input_matrix (np.array): the input vector

        Returns:
            np.array: the predicted output vector
        """
        output = input_matrix
        for i in range(1, len(self.layers) - 1):
            weighted_sum = np.dot(output, self.matrices[f'Weight{i}']) + self.matrices[f'Bias{i}']
            output = self.threshold_function(weighted_sum)
            
        # Output layer, might have different activation function
        weighted_sum = np.dot(output, self.matrices[f'Weight{len(self.layers) - 1}']) + self.matrices[f'Bias{len(self.layers) - 1}']
        output = self.output_function(weighted_sum)

        # # Test output
        # print(output)

        return output

    def forward_propagation(self, input_matrix):
        """
        Produce the layer results for backpropagation

        Args:
            input_matrix (np.array): the input vector

        Returns:
            np.array(np.array): the results coming from each layer during the prediction
        """
        layer_res=[input_matrix]
        output = input_matrix
        for i in range(1, len(self.layers) - 1):
            weighted_sum = np.dot(output, self.matrices[f'Weight{i}']) + self.matrices[f'Bias{i}']
            output = self.threshold_function(weighted_sum)
            layer_res.append(output)
            
        # Output layer, might have different activation function
        weighted_sum = np.dot(output, self.matrices[f'Weight{len(self.layers) - 1}']) + self.matrices[f'Bias{len(self.layers) - 1}']
        output = self.output_function(weighted_sum)
        layer_res.append(output)
        
        return layer_res

    def backpropagation(self, expectation, layer_results):
        """
        Updates model weights using backpropagation algorithm

        Args:
            expectation (np.array): the expect-to-be-predicted vector
            layer_results (np.array(np.array)): the results coming from each layer during the prediction
        """
        loss = self.d_loss(layer_results[-1], expectation)
        
        i = len(self.layers) - 1
        # Output Layer, might have different activation function
        corrected_loss = self.d_output(layer_results[i]) * loss
        self.matrices[f'Weight{i}'] = self.matrices[f'Weight{i}'] - self.learn_rate * (np.dot(np.atleast_2d(layer_results[i-1]).T, np.atleast_2d(corrected_loss)))
        self.matrices[f'Bias{i}'] = self.matrices[f'Bias{i}'] - self.learn_rate * corrected_loss
        loss = np.dot(corrected_loss, self.matrices[f'Weight{i}'].T)
        
        for i in range(len(layer_results)-2, 0, -1):
            corrected_loss = self.d_threshold(layer_results[i]) * loss
            self.matrices[f'Weight{i}'] = self.matrices[f'Weight{i}'] - self.learn_rate * (np.dot(np.atleast_2d(layer_results[i-1]).T, np.atleast_2d(corrected_loss)))
            self.matrices[f'Bias{i}'] = self.matrices[f'Bias{i}'] - self.learn_rate * corrected_loss
            loss = np.dot(corrected_loss, self.matrices[f'Weight{i}'].T)

    def train(self, data, target):
        """
        Train the model using training dataset

        Args:
            data (np.array(np.array)): A list of feature vectors that contains features which can capture the characteristics of data.
            target (np.array): A list of integer that indicate the class an according feature vector is in.
        """
        for _ in range(self.step):
            for feature, expectation in zip(data, target):
                # Forward Propagation to obtain results in layers
                layer_res = self.forward_propagation(feature)
                # Backward Propagation to change the weights
                self.backpropagation(expectation, layer_res)
                # # Test training results
                # print(self.matrices)