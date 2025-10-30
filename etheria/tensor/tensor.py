from etheria._etheria import Tensor as _CTensor
from typing import List, Union, Sequence, Tuple


class Tensor:
    """
    High-level Python Tensor wrapper for the underlying C++ Tensor.
    Supports multi-dimensional indexing, arithmetic operations, and
    optional requires_grad flag for autograd.
    """

    def __init__(
        self,
        data: Union[float, Sequence, None] = None,
        shape: Union[Sequence[int], None] = None,
        device_id: int = -1,
        requires_grad: bool = False,
    ):
        self.requires_grad = requires_grad

        self._tensor: _CTensor

        if isinstance(data, float):
            # Scalar tensor
            self._tensor = _CTensor(data=data, requires_grad=requires_grad, device_id=device_id)
        elif isinstance(data, Sequence):
            if shape is None:
                # Infer shape as 1D tensor
                self._tensor = _CTensor(data=Tensor.flat_sequence(data), requires_grad=requires_grad, device_id=device_id)
            else:
                self._tensor = _CTensor(data=Tensor.flat_sequence(data), shape=list(shape), requires_grad=requires_grad, device_id=device_id)
        elif data is None:
            if shape is None:
                raise ValueError("Either data or shape must be provided")
            self._tensor = _CTensor(shape=list(shape), requires_grad=requires_grad, device_id=device_id)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    @staticmethod
    def flat_sequence(seq: Sequence) -> Sequence[float]:
        """
        Given a sequence (possibly nested), flatten it into a flat list of floats. 
        """
        flat_list: list = []
        for item in seq:
            if isinstance(item, (int, float)):
                flat_list.append(float(item))
            elif isinstance(item, Sequence):
                flat_list.extend(Tensor.flat_sequence(item))
            else:
                raise TypeError(f"Unsupported data type in sequence: {type(item)}")
        return flat_list

    def __getitem__(self, idx: Union[int, Tuple[int, ...]]) -> "Tensor":
        if isinstance(idx, int):
            sub_ctensor = self._tensor.get_subtensor(idx)
        else:
            sub_ctensor = self._tensor.get_subtensor_multi(list(idx))

        # Always return a Tensor
        return Tensor._from_ctensor(sub_ctensor)

    def __setitem__(self, idx: Union[int, Tuple[int, ...]], value: Union[float, "Tensor"]) -> None:
        if isinstance(idx, int):
            idx = (idx,)
        self._tensor.set_flat_multi(list(idx), value)

    # Arithmetic operators
    def __add__(self, other: Union[float, "Tensor"]) -> "Tensor":
        if isinstance(other, (int, float)):
            result: _CTensor = self._tensor.add_scalar(float(other))
            return Tensor._from_ctensor(result)
            
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported type for addition: {type(other)}")
        
        if self.shape != other.shape:
            raise ValueError(f"Tensors must have the same shape for addition. Self shape: {self.shape}, Other shape: {other.shape}")

        result: _CTensor = self._tensor.add(other._tensor)
        return Tensor._from_ctensor(result)

    def __sub__(self, other: Union[float, "Tensor"]) -> "Tensor":
        if isinstance(other, (int, float)):
            other = Tensor(data=float(other), requires_grad=False)
        result: _CTensor = self._tensor.subtract(other._tensor)
        return Tensor._from_ctensor(result)

    def __mul__(self, other: Union[float, "Tensor"]) -> "Tensor":
        """
        Element-wise multiplication
        """
        if isinstance(other, (int, float)):
            result: _CTensor = self._tensor.scalar_multiply(float(other))
            return Tensor._from_ctensor(result)
        
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported type for multiplication: {type(other)}")
        
        if self.shape != other.shape:
            raise ValueError(f"Tensors must have the same shape for element-wise multiplication. Self shape: {self.shape}, Other shape: {other.shape}")
        
        result: _CTensor = self._tensor.multiply(other._tensor)
        return Tensor._from_ctensor(result)
    
    def __div__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        if isinstance(other, (int, float)):
            if other == 0:
                raise ValueError("Division by zero")
            scalar = 1.0 / float(other)
            result: _CTensor = self._tensor.scalar_multiply(scalar)
            return Tensor._from_ctensor(result)
        else:
            raise TypeError("Division only supported with scalars")

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, (Tensor, int, float)):
            return False
        if isinstance(value, (int, float)):
            value = Tensor(data=float(value), requires_grad=False)

        if self.shape != value.shape:
            return False
        
        return self.to_list() == value.to_list()
    
    def outer_product(self, other: "Tensor") -> "Tensor":
        result: _CTensor = self._tensor.outer_product(other._tensor)
        return Tensor._from_ctensor(result)
    
    def dot_product(self, other: "Tensor") -> "Tensor":
        result: _CTensor = self._tensor.dot_product(other._tensor)
        return Tensor._from_ctensor(result)

    def transpose(self, axes: List[int] = []) -> "Tensor":
        if not axes:
            axes = list(range(len(self.shape)))[::-1]  # reverse order

        if len(axes) != len(self.shape):
            raise ValueError("Axes length must match tensor dimensions")
        
        result: _CTensor = self._tensor.transpose(axes)
        return Tensor._from_ctensor(result)

    def to_list(self) -> list:
        # since pybind11 binds a C++ std::vector to a Python list directly we can return it as is
        vector: list = self._tensor.to_vector()

        # now we have a vector of the flat data, we need to reshape it to the tensor shape
        def reshape(data: list, shape: Tuple[int, ...]) -> Union[float, list]:
            if len(shape) == 0:
                return data[0]  # scalar
            if len(shape) == 1:
                return data[: shape[0]]  # last dimension
            size = shape[0]
            step = int(len(data) / size)
            return [reshape(data[i * step : (i + 1) * step], shape[1:]) for i in range(size)]

        return reshape(vector, self.shape)

    # Shape and device info
    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._tensor.get_shape())
    
    @property
    def rank(self) -> int:
        return self._tensor.get_rank()

    def reshape(self, new_shape: Union[Tuple[int, ...], Sequence[int]]) -> "Tensor":
        return Tensor._from_ctensor(self._tensor.reshape(list(new_shape)))

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

    # Internal constructor from cpp Tensor
    @classmethod
    def _from_ctensor(cls, ctensor: _CTensor):
        obj: Tensor = cls.__new__(cls)
        obj._tensor = ctensor
        obj.requires_grad = False
        return obj

    @classmethod
    def from_list(
        cls,
        data: Union[float, Sequence[float]],
        shape: Union[Tuple[int, ...], Sequence[int]],
        requires_grad: bool = False,
    ) -> "Tensor":
        return cls(data=data, shape=shape, requires_grad=requires_grad)

    def __del__(self) -> None:
        del self._tensor

    # Representation
    def __repr__(self):
        t_type = "Tensor"
        if len(self.shape) == 0:
            t_type = "Escalar"
        if len(self.shape) == 1:
            t_type = "Vector"
        elif len(self.shape) == 2:
            t_type = "Matriz"

        return f"Tensor({t_type})(shape={self.shape} rank={self.rank} data={self.to_list()}) at device {self.device} at {hex(id(self))}"
