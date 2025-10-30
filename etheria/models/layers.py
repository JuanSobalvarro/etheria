from etheria.tensor import Tensor
from etheria.activation import ActivationFunction, apply_activation, apply_derivative
from typing import Union

class Layer:
    """
    Base class for all layers in a MLP. It contains just the metadata, we do not operate over the data here.
    """

    def __init__(self, neurons: int, activation: Union[ActivationFunction, str], **kwargs):
        self.neurons: int = neurons
        self.Z: Tensor | None = None  # Pre-activation values
        self.A: Tensor | None = None  # Post-activation values
        # Normalize activation to an ActivationFunction enum
        self.activation: ActivationFunction = ActivationFunction.from_value(activation)
        self.kwargs = kwargs

    @property
    def pre_activation(self) -> Tensor:
        if self.Z is None:
            raise ValueError("Pre-activation values have not been computed yet.")
        return self.Z
    
    @property
    def output(self) -> Tensor:
        if self.A is None:
            raise ValueError("Post-activation values have not been computed yet.")
        return self.A

    def forward(self, inputs: Tensor, weights: Tensor, biases: Tensor) -> Tensor:
        """
        Forward pass through the layer. To be implemented by subclasses.
        """
        raise NotImplementedError("Forward method must be implemented by subclasses.")

class DenseLayer(Layer):
    """
    This class represents a dense layer in a MLP. It contains just the metadata, we do not operate over the data here.
    """

    def __init__(self, neurons: int, activation: Union[ActivationFunction, str], **kwargs):
        super().__init__(neurons, activation, **kwargs)

    def forward(self, inputs: Tensor, weights: Tensor, biases: Tensor) -> Tensor:
        """
        Forward pass through the dense layer.
        Z = W·X + b
        A = activation(Z)

        inputs: Tensor of shape (input_dim,)
        weights: Tensor of shape (neurons, input_dim)
        biases: Tensor of shape (neurons,)
        """ 

        # check ranks
        if inputs.rank != 1:
            raise ValueError(f"Inputs must be a 1D tensor, got rank {inputs.rank}")
        
        if weights.rank != 2:
            raise ValueError(f"Weights must be a 2D tensor, got rank {weights.rank}")
        
        if biases.rank != 1:
            raise ValueError(f"Biases must be a 1D tensor, got rank {biases.rank}")


        if inputs.shape != (weights.shape[1],):
            raise ValueError(f"Input shape {inputs.shape} does not match expected shape {(weights.shape[1],)}")

        # Linear transformation
        prod = weights.outer_product(inputs)
        print(f"Weight shape: {weights.shape}, Input shape: {inputs.shape}, Prod shape: {prod.shape}, Biases shape: {biases.shape}")
        self.Z = prod + biases
        # Apply activation function
        self.A = apply_activation(self.Z, self.activation)
        return self.A
    
    def clear_cache(self):
        """
        Clear cached values to free memory.
        """
        self.Z = None
        self.A = None
    
    def __repr__(self) -> str:
        return f"DenseLayer(neurons={self.neurons}, activation={self.activation.name})"
