import warnings
import numpy as np
from .NN import NeuralNetwork

class Classifier:
    """
    Classifier interface to use the NN.

    Built for classification task with training data structure:
        data: List[List[float]]
            [feature_vector1, feature_vector2, ...]
        target: List[int]
            The class of the according feature_vector in data.
        feature_vector: List[float]
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
        self.net.output_function = self.net.threshold_function
        self.net.d_output = self.net.d_threshold

    def __init__(self, net: NeuralNetwork):
        """
        Args:
            net (NeuralNetwork): Raw configured NeuralNetwork.
        """
        self.layers = net.layers
        self.net = net

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
