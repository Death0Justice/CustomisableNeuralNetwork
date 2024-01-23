"""
This module provides interfaces to use a layer-customizable
Neural Network.

Classes:
    Classifier: A classifier interface adopting the NN.
    
    Regressor: A regressor interface adopting the NN.
    
    NeuralNetwork: The raw neural network
"""

__version__ = "0.1.0"
__author__ = "Death Justice"
__email__ = "death0justice@gmail.com"

from .classifier import Classifier
from .regressor import Regressor
from .NN import NeuralNetwork
from .functions import *