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

def Tanh(x):
    """
    Hyperbolic Tangent Function
    Can be used as a threshold function below

    Parameters:
    ----------
    x: float
        The input of Tanh function

    Returns:
    -------
    return_value: float
        The output of Tanh function
    """
    return 2.0 / (1.0 + np.exp(-2 * x)) - 1

# Note that y is the Tanh output
def dTanh(y):
    """
    Derivative of Tanh Function

    Parameters:
    ----------
    y: float
        The *output* of Tanh function

    Returns:
    -------
    return_value: float
        The derivative of Tanh function when output is y
    """
    return 1.0 - y * y

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

def Linear(x):
    """
    Linear Function
    Can be used as a output activation function below

    Parameters:
    ----------
    x: float
        The input of Linear function

    Returns:
    -------
    return_value: float
        The output of Linear function
    """
    return x

# Note that y is the Linear output
def dLinear(y):
    """
    Derivative of Linear Function

    Parameters:
    ----------
    y: float
        The *output* of Linear function

    Returns:
    -------
    return_value: float
        The derivative of Linear function when output is y
    """
    return 1

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