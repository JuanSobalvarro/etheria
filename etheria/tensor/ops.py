from etheria.tensor.tensor import Tensor

def add(a: Tensor, b: Tensor) -> Tensor:
    """
    Element-wise addition of two tensors.

    Args:
        a (Tensor): The first tensor.
        b (Tensor): The second tensor.

    Returns:
        Tensor: The result of element-wise addition.
    """
    # sanity check for shapes
    if a.shape != b.shape:
        raise ValueError(f"Shapes of tensors do not match for addition: {a.shape} vs {b.shape}")

    return a + b

def matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    Matrix multiplication of two tensors. The tensors must be 2-dimensional (rank-2).

    Args:
        a (Tensor): The first tensor.
        b (Tensor): The second tensor.

    Returns:
        Tensor: The result of matrix multiplication.
    """
    # sanity check for shapes
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise ValueError("Both tensors must be 2-dimensional for matrix multiplication")

    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes for matrix multiplication: {a.shape} and {b.shape}")

    result_shape = (a.shape[0], b.shape[1])
    result = Tensor(shape=result_shape)

    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            sum_value = 0.0
            for k in range(a.shape[1]):
                sum_value += a[i, k] * b[k, j]
            result[i, j] = sum_value

    return result