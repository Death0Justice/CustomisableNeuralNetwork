"""
This module provides interfaces to use a layer-customizable
Neural Network.

Classes:
    Classifier: A classifier interface adopting the NN.
"""

import warnings
import numpy as np

def ReLU(x):
    """
    Rectified Linear Unit Function
    Can be used as a threshold function below

    Parameters:
    ----------
    x: float
        The input of ReLU function

    Returns:
    -------
    return_value: float
        The output of ReLU function
    """
    return x * (x > 0)

# Note that y is the ReLU output
def dReLU(y):
    """
    Derivative of ReLU Function

    Parameters:
    ----------
    y: float
        The *output* of ReLU function

    Returns:
    -------
    return_value: float
        The derivative of ReLU function when output is y
    """
    return 1 * (y > 0)

def Sigmoid(x):
    """
    Sigmoid Function
    Can be used as a threshold function below

    Parameters:
    ----------
    x: float
        The input of Sigmoid function

    Returns:
    -------
    return_value: float
        The output of Sigmoid function
    """
    return 1.0 / (1.0 + np.exp(-x))

# Note that y is the Sigmoid output
def dSigmoid(y):
    """
    Derivative of Sigmoid Function

    Parameters:
    ----------
    y: float
        The *output* of Sigmoid function

    Returns:
    -------
    return_value: float
        The derivative of Sigmoid function when output is y
    """
    return y * (1.0 - y)

def epsilon(x):
    """
    Epsilon to make x never be exactly zero
    Can be used in log function

    Parameters:
    ----------
    x: float
        The input variable that may be zero

    Returns:
    -------
    return_value: float
        The input variable that will not be zero
    """
    eps = 0.0000001
    if x > eps:
        return x
    else:
        return eps

def CrossEntropy(output, expectation):
    """
    Cross Entrophy Function
    Can be used as a loss function below

    Parameters:
    ----------
    output: float
        The output of the neurons
    expectation: float
        The expected output of the neurons

    Returns:
    -------
    return_value: float
        The loss output of Cross Entropy function
    """
    output_inv = 1.0 - output
    expectation_inv = 1.0 - expectation
    output = epsilon(output)
    output_inv = epsilon(output_inv)
    return -np.sum(np.multiply(np.log(output), expectation) \
            + np.multiply(np.log(output_inv), (expectation_inv)))

def dCrossEntropy(output, expectation):
    """
    Derivative of Cross Entropy Function

    Parameters:
    ----------
    output: float
        The output of the neurons
    expectation: float
        The expected output of the neurons

    Returns:
    -------
    return_value: float
        The derivative of Cross Entropy function
    """
    return -expectation / output + (1.0 - expectation) / (1.0 - output)

def SquareError(output, expectation):
    """
    Squared Error Function
    Can be used as a loss function below

    Parameters:
    ----------
    output: float
        The output of the neurons
    expectation: float
        The expected output of the neurons

    Returns:
    -------
    return_value: float
        The loss output of Squared Error function
    """
    return 0.5 * np.square(expectation - output)

# Derivative of the Squared Error Function
def dSquareError(output, expectation):
    """
    Derivative of Cross Entropy Function

    Parameters:
    ----------
    output: float
        The output of the neurons
    expectation: float
        The expected output of the neurons

    Returns:
    -------
    return_value: float
        The derivative of Squared Error function
    """
    return output - expectation

class Classifier:
    """
    Classifier interface to use the NN.

    Built for classification task with training data structure:
        data: List[List[int]]
            [feature_vector1, feature_vector2, ...]
        target: List[int]
            The class of the according feature_vector in data.
        feature_vector: List[int]
            A vector of features that can capture data characteristics.
    
    Attributes:
        layers: List[int]
            A list of node number of the layers of NN, 
            with the following sequence:
                [
                    input_layer,
                    hidden_layer1,
                    hidden_layer2,
                    ...,
                    output_layer
                ]
        net:
            The generated and ready-to-train Neural Network.
    """

    def __init__(self, layers):
        """
        Args:
            layers (List[Int]): Specified custom layer configuration.
        """
        self.layers = layers
        self.net = NeuralNetwork(layers=self.layers)

    def reset(self):
        """
        Reset the Neural Network to fresh state.
        """
        self.net = NeuralNetwork(layers=self.layers)

    def fit(self, data, target):
        """
        Fit the training dataset into the Neural Network.

        Args:
            data: 
                A list of feature vectors that contains features which can capture the characteristics of data.
            target:
                A list of integer that indicate the class an according feature vector is in.

        Raises:
        ExceptionType: ValueError
            When specified output layer has less nodes than target
            classes.
        """
        may_not_necessary = True
        neural_targets = []
        for target_i in target:
            if target_i == self.layers[-1] - 1:
                may_not_necessary = False
            elif target_i >= self.layers[-1]:
                raise ValueError(f"Target({target_i + 1}) has more classes than specified output nodes({self.layers[-1]}).")
            neural_target = np.zeros(self.layers[-1])
            neural_target[target_i] = 1
            neural_targets.append(neural_target)
        if may_not_necessary is True:
            warnings.warn(f"Target({np.max(target)}) has less classes than specified output layer({self.layers[-1]}), may incur unnecessary model complexity.", RuntimeWarning)
        self.net.train(data, neural_targets)

    def predict(self, data):
        """
        Predict the class of a given feature vector

        Args:
            data: 
                A feature vector that can capture the characteristics
                of data.
        """
        print(np.argmax(self.net.predict(data)))
        return np.argmax(self.net.predict(data))

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
        self.matrices = {}
        self.init_matrices()

    def init_matrices(self):
        """
        Initialize the matrices using Normal Distribution randomize
        """
        for i in range(1, len(self.layers)):
            self.matrices[f'Weight{i}'] = np.random.randn(self.layers[i-1], self.layers[i])
            self.matrices[f'Bias{i}'] = np.random.randn(self.layers[i],)

    def threshold_function(self, weighted_input):
        """
        The threshold function in the NN.

        Args:
            weighted_input (float): the weighted sum of neurons output from previous layer.

        Returns:
            float: the calcuated value to be compared with threshold
        """
        return Sigmoid(weighted_input)

    # Note that the parameter is the output of that function
    def d_threshold(self, output):
        """
        The derivative of the threshold function

        Args:
            output (float): the *output* of the threshold function

        Returns:
            float: the derivative of the threshold function
        """
        return dSigmoid(output)

    def loss_function(self, output, expectation):
        """
        The loss function
        Can be used to plot loss curve

        Args:
            output (float): the output of a neuron
            expectation (float): the expected output of a neuron

        Returns:
            float: the loss
        """
        return SquareError(output, expectation)

    def d_loss(self, output, expectation):
        """
        The derivative of the loss function

        Args:
            output (float): the output of a neuron
            expectation (float): the expected output of a neuron

        Returns:
            float: the derivative of the loss function
        """
        return dSquareError(output, expectation)


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
        for i in range(1, len(self.layers)):
            weighted_sum = np.dot(output, self.matrices[f'Weight{i}']) + self.matrices[f'Bias{i}']
            output = self.threshold_function(weighted_sum)

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
        for i in range(1, len(self.layers)):
            weighted_sum = np.dot(output, self.matrices[f'Weight{i}']) + self.matrices[f'Bias{i}']
            output = self.threshold_function(weighted_sum)
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
        for i in range(len(layer_results)-1, 0, -1):
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
