import numpy as np
from typing import List
from etheria.tensor import Tensor
from etheria.activation import ActivationFunction


def initialize_he(weights: Tensor, biases: Tensor, fan_in: int, fan_out: int):
    """
    Initialize weights and biases using He initialization.

    He initialization is particularly suited for layers with ReLU activation functions.

    Given weights of shape (num_neurons, num_inputs) and biases of shape (num_neurons,),
    this function initializes them in place. 
    """

    if weights.shape[0] != fan_out or weights.shape[1] != fan_in:
        raise ValueError(f"Weights shape: {weights.shape} does not match the provided fan_in {fan_in} and fan_out {fan_out} values.")

    stddev = np.sqrt(2.0 / fan_in)

    # since every neuron is fully connected to all inputs, we can just fill the weights and biases directly using shape[0] and shape[1]

    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            weights[i][j] = np.random.randn() * stddev

    for i in range(biases.shape[0]):
        biases[i] = 0.0


def initialize_normal(weights: Tensor, biases: Tensor, fan_in: int, fan_out: int):
    """
    Initialize weights and biases using a normal distribution.

    This is a generic initialization method that can be used for layers with linear activation functions.

    Given weights of shape (num_neurons, num_inputs) and biases of shape (num_neurons,),
    this function initializes them in place. 
    """

    if weights.shape[0] != fan_out or weights.shape[1] != fan_in:
        raise ValueError("Weights shape does not match the provided fan_in and fan_out values.")

    stddev = 0.01  # A small standard deviation for normal initialization

    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            weights[i][j] = np.random.randn() * stddev

    for i in range(biases.shape[0]):
        biases[i] = 0.0


def do_init(weights, biases, fan_in, fan_out, activation: ActivationFunction):
    if activation == ActivationFunction.RELU:
        initialize_he(weights, biases, fan_in, fan_out)
    elif activation == ActivationFunction.LINEAR:
        initialize_normal(weights, biases, fan_in, fan_out)
    elif activation == ActivationFunction.SIGMOID:
        initialize_normal(weights, biases, fan_in, fan_out)
    elif activation == ActivationFunction.TANH:
        initialize_normal(weights, biases, fan_in, fan_out)
    else:
        raise NotImplementedError(f"Initialization for activation {activation} is not implemented yet.")
