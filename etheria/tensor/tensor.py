import numpy as np
from etheria.core import _Tensor


# TODO: Implement internal tensor class in C++ and wrap it here
# For now, we just create a dummy class to avoid errors
class Tensor:
    """
    A eth.Tensor is a multi-dimensional array used for numerical computations. It supports dtype values inside of it
    and can be created from numpy arrays or other tensor-like structures.
    """
    def __init__(self, data, dtype=None, device="cpu"):
        if isinstance(data, np.ndarray):
            self._tensor = _Tensor.from_numpy(data, dtype, device)
        elif isinstance(data, _Tensor):
            self._tensor = data
        else:
            raise TypeError("Unsupported input type for Tensor")

    @property
    def shape(self):
        return self._tensor.shape()

    @property
    def device(self):
        return self._tensor.device()

    def numpy(self):
        return self._tensor.to_numpy()

    def to(self, device: str):
        return Tensor(self._tensor.to(device))

    def __repr__(self):
        return f"Tensor(shape={self.shape}, device={self.device})"
