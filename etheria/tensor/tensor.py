from etheria._etheria import Tensor as _CTensor
from typing import Union, Sequence, Tuple


class Tensor:
    """
    High-level Python Tensor wrapper for the underlying C++ Tensor.
    Supports multi-dimensional indexing, arithmetic operations, and
    optional requires_grad flag for autograd.
    """

    def __init__(
        self,
        data: Union[float, Sequence[float], None] = None,
        shape: Union[Sequence[int], None] = None,
        requires_grad: bool = False,
    ):
        self.requires_grad = requires_grad

        self._tensor: _CTensor

        if isinstance(data, float):
            # Scalar tensor
            self._tensor = _CTensor(data=data, requires_grad=requires_grad)
        elif isinstance(data, list):
            if shape is None:
                # Infer shape as 1D tensor
                self._tensor = _CTensor(data=data, requires_grad=requires_grad)
            else:
                self._tensor = _CTensor(data=data, shape=list(shape), requires_grad=requires_grad)
        elif data is None:
            if shape is None:
                raise ValueError("Either data or shape must be provided")
            self._tensor = _CTensor(shape=list(shape), requires_grad=requires_grad)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    # Indexing
    def __getitem__(self, idx: Union[int, Tuple[int, ...]]):
        if isinstance(idx, int):
            idx = (idx,)
        return self._tensor.get(list(idx))

    def __setitem__(self, idx: Union[int, Tuple[int, ...]], value: float):
        if isinstance(idx, int):
            idx = (idx,)
        self._tensor.set(list(idx), value)

    # Arithmetic operators
    def __add__(self, other: "Tensor") -> "Tensor":
        result: _CTensor = self._tensor.add(other._tensor)
        return Tensor._from_ctensor(result)

    def __mul__(self, other: "Tensor") -> "Tensor":
        result: _CTensor = self._tensor.multiply(other._tensor)
        return Tensor._from_ctensor(result)
    
    def outer_product(self, other: "Tensor") -> "Tensor":
        result: _CTensor = self._tensor.outer_product(other._tensor)
        return Tensor._from_ctensor(result)
    
    def dot_product(self, other: "Tensor") -> "Tensor":
        result: _CTensor = self._tensor.dot_product(other._tensor)
        return Tensor._from_ctensor(result)

    def to_list(self) -> list:
        # since pybind11 binds a C++ std::vector to a Python list directly we can return it as is
        return self._tensor.to_vector()

    # Shape and device info
    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._tensor.get_shape())

    @property
    def num_elements(self) -> int:
        return self._tensor.size()

    @property
    def device(self) -> int:
        return self._tensor.current_device()

    # Move between CPU/GPU
    def to_cpu(self):
        self._tensor.move_to_cpu()

    def to_gpu(self, device_id: int = 0):
        self._tensor.move_to_gpu(device_id)

    # Internal constructor from C++ Tensor
    @classmethod
    def _from_ctensor(cls, ctensor: _CTensor):
        obj: Tensor = cls.__new__(cls)
        obj._tensor = ctensor
        obj.requires_grad = False
        return obj

    # Representation
    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.to_list()})"
