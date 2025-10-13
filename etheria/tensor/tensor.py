from etheria._eth import Tensor as _Tensor
from etheria._eth import add as _add, matmul as _matmul, activation as _activation


class Tensor:
    def __init__(self, shape, data=None, tensor: _Tensor = None):
        self.dims = len(shape)
        if tensor is not None:
            self._t = tensor
        else:
            self._t = _Tensor(shape)
        if data is not None:
            self._t.data = self.flatten_sequence(data)

    @staticmethod
    def flatten_sequence(seq):
        """Flattens a nested sequence into a single list of floats"""
        if not isinstance(seq, (list, tuple)):
            # unwrap TensorView if needed
            if hasattr(seq, "data"):
                seq = seq.data
            return [float(seq)]
        result = []
        for item in seq:
            if hasattr(item, "data"):
                # unwrap TensorView
                item = item.data
            result.extend(Tensor.flatten_sequence(item))
        return result

    @property
    def shape(self):
        return tuple(self._t.shape)

    @property
    def data(self):
        def nest(data, shape):
            if len(shape) == 1:
                return data
            step = len(data) // shape[0]
            return [nest(data[i*step:(i+1)*step], shape[1:]) for i in range(shape[0])]
        return nest(list(self._t.data), self._t.shape)

    def _flatten_index(self, indices):
        flat_index = 0
        for i, idx in enumerate(indices):
            flat_index = flat_index * self.shape[i] + idx
        return flat_index

    def __getitem__(self, index):
        if isinstance(index, int):
            return TensorView(self, [index])
        elif isinstance(index, tuple):
            return TensorView(self, list(index))
        elif isinstance(index, slice):
            indices = list(range(*index.indices(self.shape[0])))
            return [TensorView(self, [i]) for i in indices]
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def __setitem__(self, index, value):
        flat_index = self._flatten_index([index])
        self._t[flat_index] = value

    def __iter__(self):
        for i in range(self.shape[0]):
            yield TensorView(self, [i])

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data})"


class TensorView:
    """
    Provides chained indexing into a Tensor without copying data.
    """
    def __init__(self, tensor, indices):
        self.tensor = tensor
        self.indices = indices

    def __getitem__(self, index):
        new_indices = self.indices + ([index] if isinstance(index, int) else list(index))
        if len(new_indices) == self.tensor.dims:
            flat_index = self.tensor._flatten_index(new_indices)
            return self.tensor._t[flat_index]
        return TensorView(self.tensor, new_indices)

    def __setitem__(self, index, value):
        flat_index = self.tensor._flatten_index(self.indices + [index])
        self.tensor._t[flat_index] = value

    def __iter__(self):
        remaining_shape = self.tensor.shape[len(self.indices):]
        if not remaining_shape:
            raise TypeError("Cannot iterate over a scalar")
        for i in range(remaining_shape[0]):
            yield TensorView(self.tensor, self.indices + [i])

    @property
    def data(self):
        remaining_shape = self.tensor.shape[len(self.indices):]
        if not remaining_shape:
            return self.tensor._t[self.tensor._flatten_index(self.indices)]

        def recursive_data(indices, shape):
            if len(shape) == 1:
                return [self.tensor._t[self.tensor._flatten_index(indices + [i])] for i in range(shape[0])]
            return [recursive_data(indices + [i], shape[1:]) for i in range(shape[0])]

        return recursive_data(self.indices, remaining_shape)

    @property
    def as_tensor(self):
        # return data as a Tensor object
        data = self.data
        shape = self.tensor.shape[len(self.indices):]
        return Tensor(shape=shape, data=data)

    def __repr__(self):
        return f"TensorView(indices={self.indices}, data={self.data})"


def add(a: Tensor, b: Tensor) -> Tensor:
    # Promote 1D to column if needed
    if len(b.shape) == 1 and len(a.shape) == 2 and a.shape[1] == 1:
        b = Tensor(shape=(b.shape[0], 1), data=b.data)
    
    if a.shape != b.shape:
        raise ValueError(f"Shapes must match for addition: {a.shape} vs {b.shape}")
    
    return Tensor(a.shape, tensor=_add(a._t, b._t))


def matmul(a: Tensor, b: Tensor) -> Tensor:
    a_t = a._t
    b_t = b._t

    # reshape 1D vectors to 2D for C++
    if len(a.shape) == 1:
        a_t = _Tensor([1, a.shape[0]])
    if len(b.shape) == 1:
        b_t = _Tensor([b.shape[0], 1])

    result_t = _matmul(a_t, b_t)

    # determine final shape
    if len(a.shape) == 1 and len(b.shape) == 1:
        return Tensor(shape=(), tensor=result_t)  # scalar
    elif len(a.shape) == 1:
        return Tensor(shape=(b.shape[0],), tensor=result_t)
    elif len(b.shape) == 1:
        return Tensor(shape=(a.shape[0],), tensor=result_t)
    else:
        return Tensor(shape=(a.shape[0], b.shape[1]), tensor=result_t)


def activation(input: Tensor, activation: str) -> Tensor:
    result = Tensor(input.shape)
    result._t = _activation(input._t, activation)
    return result