from etheria.activation import ActivationFunction

class Layer:
    """
    Base class for all layers in a MLP. It contains just the metadata, we do not operate over the data here.
    """

    def __init__(self, neurons: int, activation: ActivationFunction, **kwargs):
        self.neurons: int = neurons
        self.activation: ActivationFunction = activation
        self.kwargs = kwargs

class DenseLayer(Layer):
    """
    This class represents a dense layer in a MLP. It contains just the metadata, we do not operate over the data here.
    """

    def __init__(self, neurons: int, activation: ActivationFunction, **kwargs):
        super().__init__(neurons, activation, **kwargs)
