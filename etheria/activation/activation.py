from typing import Union
from enum import Enum
from etheria._etheria import ActivationFunction as _CActivationFunction
from etheria._etheria import activation as _activation
from etheria._etheria import activation_derivative as _activation_derivative
from etheria.tensor import Tensor


class ActivationFunction(Enum):
    LINEAR = _CActivationFunction.LINEAR
    RELU = _CActivationFunction.RELU
    SIGMOID = _CActivationFunction.SIGMOID
    TANH = _CActivationFunction.TANH

    @staticmethod
    def from_value(value: Union["ActivationFunction", str]) -> "ActivationFunction":
        if isinstance(value, ActivationFunction):
            return value

        if isinstance(value, str):
            value = value.strip().lower()
            mapping = {
                "linear": ActivationFunction.LINEAR,
                "relu": ActivationFunction.RELU,
                "sigmoid": ActivationFunction.SIGMOID,
                "tanh": ActivationFunction.TANH,
            }
            if value in mapping:
                return mapping[value]
            else:
                raise ValueError(f"Unknown activation function '{value}'")

        raise TypeError(f"Unsupported activation type: {type(value)}")

def apply_activation(input_tensor: Tensor, func: ActivationFunction):
    """
    Apply the specified activation function to the input tensor.
    """

    result_tensor = _activation(input_tensor._tensor, func.value)
    return Tensor._from_ctensor(result_tensor)

def apply_derivative(input_tensor: Tensor, func: ActivationFunction):
    """
    Apply the derivative of the specified activation function to the input tensor.
    """

    result_tensor = _activation_derivative(input_tensor._tensor, func.value)
    return Tensor._from_ctensor(result_tensor)