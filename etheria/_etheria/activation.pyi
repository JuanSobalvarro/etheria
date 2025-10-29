
from etheria._etheria.tensor import Tensor


class ActivationFunction:
    LINEAR: int
    RELU: int
    SIGMOID: int
    TANH: int

def activation(input: "Tensor", func: ActivationFunction) -> "Tensor": ...

def activation_derivative(input: "Tensor", func: ActivationFunction) -> "Tensor": ...