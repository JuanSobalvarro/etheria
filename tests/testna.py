import etheria as eth

def test_cuda():
    print("========= CUDA TESTS =========")

    print(f"The number of CUDA devices is: {eth.cuda.number_cuda_devices()}")
    print(f"The current CUDA device is: {eth.cuda.current_cuda_device()}")
    print(f"The details of CUDA device 0 are: {eth.cuda.device_details(0)}")

def test_tensor_creation():
    print("========= TENSOR CREATION TESTS =========")

    a = eth.tensor.Tensor(shape=[3, 4])
    print("Tensor a:", a)
    print("Tensor a shape:", a.shape)
    print("Tensor a size:", a.num_elements)

    b = eth.tensor.Tensor(shape=[4, 3])
    print("Tensor b:", b)
    print("Tensor b shape:", b.shape)
    print("Tensor b size:", b.num_elements)

def test_tensor_sum():
    print("========= TENSOR SUM TESTS =========")

    a = eth.tensor.Tensor(shape=[2, 2])
    b = eth.tensor.Tensor(shape=[2, 2])
    c = a + b

    print("Expected result shape: (2, 2)")
    print("Actual result shape:", c.shape)
    print("Expected result size: 4")
    print("Actual result size:", c.num_elements)

    for i in range(2):
        for j in range(2):
            expected_value = a[i, j] + b[i, j]
            actual_value = c[i, j]
            print(f"c[{i}, {j}] = {actual_value} (expected: {expected_value})")

def test_tensor_outer_product():
    print("========= TENSOR OUTER PRODUCT TESTS =========")
    a = eth.tensor.Tensor(shape=[2])
    b = eth.tensor.Tensor(shape=[3])
    c = a.outer_product(b)
    print("Expected result shape: (2, 3)")
    print("Actual result shape:", c.shape)
    print("Expected result size: 6")
    print("Actual result size:", c.num_elements)

    for i in range(2):
        for j in range(3):
            expected_value = a[i] * b[j]
            actual_value = c[i, j]
            print(f"c[{i}, {j}] = {actual_value} (expected: {expected_value})")

def test_tensor_population():
    print("========= TENSOR POPULATION TESTS =========")
    a = eth.tensor.Tensor(shape=[2, 2])
    print("Populating tensor a:")
    for i in range(2):
        for j in range(2):
            value = float(i * 2 + j + 1)
            a[i, j] = value
            print(f"Set a[{i}, {j}] = {value}")

    print("Verifying tensor a:")
    for i in range(2):
        for j in range(2):
            value = a[i, j]
            expected_value = float(i * 2 + j + 1)
            print(f"a[{i}, {j}] = {value} (expected: {expected_value})")

    b = eth.tensor.Tensor(shape=[2, 2])
    for i in range(2):
        for j in range(2):
            value = float((i + 1) * (j + 1))
            b[i, j] = value

    c = a + b
    print("Verifying tensor c = a + b:")
    for i in range(2):
        for j in range(2):
            expected_value = a[i, j] + b[i, j]
            actual_value = c[i, j]
            print(f"c[{i}, {j}] = {actual_value} (expected: {expected_value})")

def test_tensor_dot_product():
    print("========= TENSOR DOT PRODUCT TESTS =========")
    a = eth.tensor.Tensor(shape=[2, 3])
    b = eth.tensor.Tensor(shape=[3, 4])

    # Populate tensor a
    for i in range(2):
        for j in range(3):
            a[i, j] = float(i * 3 + j + 1)

    # Populate tensor b
    for i in range(3):
        for j in range(4):
            b[i, j] = float((i + 1) * (j + 1))

    c = a.dot_product(b)
    print("Expected result shape: (2, 4)")
    print("Actual result shape:", c.shape)
    print("Expected result size: 8")
    print("Actual result size:", c.num_elements)

    for i in range(2):
        for j in range(4):
            expected_value = sum(a[i, k] * b[k, j] for k in range(3))
            actual_value = c[i, j]
            print(f"c[{i}, {j}] = {actual_value} (expected: {expected_value})")

def main():

    test_cuda()
    test_tensor_creation()
    test_tensor_sum()
    test_tensor_outer_product()
    test_tensor_population()
    test_tensor_dot_product()

if __name__ == "__main__":
    main()