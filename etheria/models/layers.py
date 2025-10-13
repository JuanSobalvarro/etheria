from etheria.activation import ActivationFunction
from typing import Union

class Layer:
    """
    Base class for all layers in a MLP. It contains just the metadata, we do not operate over the data here.
    """

    def __init__(self, neurons: int, activation: Union[ActivationFunction, str], **kwargs):
        self.neurons: int = neurons
        # Normalize activation to an ActivationFunction enum
        if isinstance(activation, str):
            try:
                activation = ActivationFunction(activation.lower())
            except ValueError:
                raise ValueError(f"Unknown activation function: {activation}")
        elif not isinstance(activation, ActivationFunction):
            raise TypeError(f"activation must be ActivationFunction or str, got {type(activation)}")
        self.activation: ActivationFunction = activation
        self.kwargs = kwargs

class DenseLayer(Layer):
    """
    This class represents a dense layer in a MLP. It contains just the metadata, we do not operate over the data here.
    """

    def __init__(self, neurons: int, activation: Union[ActivationFunction, str], **kwargs):
        super().__init__(neurons, activation, **kwargs)
