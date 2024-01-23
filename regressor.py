from .NN import NeuralNetwork

class Regressor:
    """
    Regressor interface to use the NN.

    Built for regression task with training data structure:
        data: List[List[float]]
            [feature_vector1, feature_vector2, ...]
        target: List[float]
            The expected value of the according feature_vector in data.
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
                ]
            The output layer is implicitly added.
        net:
            The generated and ready-to-train Neural Network.
    """

    def __init__(self, layers):
        """
        Args:
            layers (List[Int]): Specified custom layer configuration.
            PS: In regressor the output layer has only one neuron, so it is defaultly put at the end of the configuration
        """
        self.layers = layers
        self.layers.append(1)
        self.net = NeuralNetwork(layers=self.layers)

    def __init__(self, net: NeuralNetwork):
        """
        Args:
            net (NeuralNetwork): Raw configured NeuralNetwork.
            PS: Due to regressor propety, the net is constructed with same configuration and an added layer
        """
        self.layers = net.layers.copy()
        self.layers.append(1)
        self.net = NeuralNetwork(layers=self.layers, learn_rate=net.learn_rate, step=net.step)

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
                A list of expected value of the given feature vector.
        """
        neural_targets = [[t] for t in target]
        self.net.train(data, neural_targets)

    def predict(self, data):
        """
        Predict the value of a given feature vector

        Args:
            data: 
                A feature vector that can capture the characteristics
                of data.
        """
        print(self.net.predict(data)[0])
        return self.net.predict(data)[0]