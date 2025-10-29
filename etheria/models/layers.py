from etheria.activation import ActivationFunction
from typing import Union

class Layer:
    """
    Base class for all layers in a MLP. It contains just the metadata, we do not operate over the data here.
    """

    def __init__(self, neurons: int, activation: Union[ActivationFunction, str], **kwargs):
        self.neurons: int = neurons
        # Normalize activation to an ActivationFunction enum
        self.activation: ActivationFunction = ActivationFunction.from_value(activation)
        self.kwargs = kwargs

class DenseLayer(Layer):
    """
    This class represents a dense layer in a MLP. It contains just the metadata, we do not operate over the data here.
    """

    def __init__(self, neurons: int, activation: Union[ActivationFunction, str], **kwargs):
        super().__init__(neurons, activation, **kwargs)
