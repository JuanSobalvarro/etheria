import etheria as eth
from etheria.models.layers import Layer
import time as t

def test_cuda():
    print("========= CUDA TESTS =========")

    print(f"The number of CUDA devices is: {eth.cuda.number_cuda_devices()}")
    print(f"The current CUDA device is: {eth.cuda.current_cuda_device()}")
    print(f"The details of CUDA device 0 are: {eth.cuda.device_details(0)}")

def test_tensor_creation():
    print("========= TENSOR CREATION TESTS =========")

    # test scalar
    itime = t.time()
    esc = eth.tensor.Tensor(data=3.14)
    etime = t.time()
    print("Scalar tensor esc:", esc)
    print(f"Time taken to create scalar tensor at cpu: {etime - itime} seconds")

    itime = t.time()
    esc.to_gpu(0)
    etime = t.time()
    print("Scalar tensor esc after moving to GPU:", esc)
    print(f"Time taken to move scalar tensor to gpu: {etime - itime} seconds")

    # currently disable cause direct creation on GPU has issues ( i am lazy to fix it now )
    # itime = t.time()
    # esc_tensor = eth.tensor.Tensor(data=2.71, device_id=0)
    # etime = t.time()
    # print("Scalar tensor esc_tensor created directly on GPU:", esc_tensor)
    # print(f"Time taken to create scalar tensor on gpu: {etime - itime} seconds")

    # a = eth.tensor.Tensor(shape=[3, 4])
    # print("Tensor a:", a)
    # print("Tensor a shape:", a.shape)
    # print("Tensor a size:", a.num_elements)

    # b = eth.tensor.Tensor(shape=[4, 3])
    # print("Tensor b:", b)
    # print("Tensor b shape:", b.shape)
    # print("Tensor b size:", b.num_elements)

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
            expected_value = a[i][j] + b[i][j]
            actual_value = c[i][j]
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
    print("========= SIMPLE TENSOR DOT PRODUCT TEST =========")
    
    # Define two small 2x2 matrices
    a = eth.tensor.Tensor(shape=[2, 2])
    b = eth.tensor.Tensor(shape=[2, 2])
    
    # Populate tensor a
    a[0, 0] = 1.0
    a[0, 1] = 2.0
    a[1, 0] = 3.0
    a[1, 1] = 4.0
    
    # Populate tensor b
    b[0, 0] = 5.0
    b[0, 1] = 6.0
    b[1, 0] = 7.0
    b[1, 1] = 8.0
    
    # Compute dot product
    c = a.dot_product(b)
    
    print("Tensor a:")
    print(a)
    print("Tensor b:")
    print(b)
    print("Dot product result c:")
    print(c)

    a_list = a.to_list()
    b_list = b.to_list()
    c_list = c.to_list()
    
    # Expected result manually computed
    # c[0,0] = 1*5 + 2*7 = 19
    # c[0,1] = 1*6 + 2*8 = 22
    # c[1,0] = 3*5 + 4*7 = 43
    # c[1,1] = 3*6 + 4*8 = 50
    expected = [[19.0, 22.0], [43.0, 50.0]]
    
    # Check each element
    for i in range(2):
        for j in range(2):
            actual = c_list[i][j]
            print(f"c[{i},{j}] = {actual} (expected: {expected[i][j]})")
            assert actual == expected[i][j], f"Mismatch at c[{i},{j}]"

def test_model_creation():
    print("========= MODEL CREATION TESTS =========")
    from etheria.models.layers import DenseLayer
    from etheria.models.sequential import SequentialModel

    layers = [
        DenseLayer(neurons=2, activation='relu'),
        DenseLayer(neurons=4, activation='sigmoid'),
        DenseLayer(neurons=1, activation='linear')
    ]

    model = SequentialModel(input_shape=(2,), layers=layers)
    print(model.summary())
    print(model.detail())

def test_activation_functions():
    print("========= ACTIVATION FUNCTION TESTS =========")
    from etheria.activation import ActivationFunction, apply_activation, apply_derivative

    a = eth.tensor.Tensor(shape=[3])
    for i in range(3):
        a[i] = float(i - 1)  # Values: -1, 0, 1

    for func in ActivationFunction:
        activated_tensor = apply_activation(a, func)
        print(f"Activation Function: {func.name}")
        for i in range(3):
            print(f"activated_tensor[{i}] = {activated_tensor[i]}")

    for func in ActivationFunction:
        derivative_tensor = apply_derivative(a, func)
        print(f"Activation Function: {func.name} (Derivative)")
        for i in range(3):
            print(f"derivative_tensor[{i}] = {derivative_tensor[i]}")

def test_model_prediction():
    print("========= MODEL PREDICTION TESTS =========")
    from etheria.models.layers import DenseLayer
    from etheria.models.sequential import SequentialModel

    layers = [
        DenseLayer(neurons=2, activation='relu'),
        DenseLayer(neurons=16, activation='sigmoid'),
        DenseLayer(neurons=128, activation='relu'),
        DenseLayer(neurons=256, activation='sigmoid'),
        DenseLayer(neurons=1, activation='linear')
    ]

    model = SequentialModel(input_shape=(2,), layers=layers)

    # Create a batch of 3 samples with input shape (2,)
    X = eth.tensor.Tensor(shape=(3, 2))
    X[0, 0] = 0.5
    X[0, 1] = -1.5
    X[1, 0] = 1.0
    X[1, 1] = 2.0
    X[2, 0] = -0.5
    X[2, 1] = 0.0

    predictions = model.predict(X)
    print("Predictions:")
    for i in range(predictions[0].shape[0] + 1):
        print(f"Sample {i}: {predictions[i]}")

def test_model_training():
    print("========= MODEL TRAINING TESTS =========")
    from etheria.models.layers import DenseLayer
    from etheria.models.sequential import SequentialModel

    # XOR problem
    layers = [
        DenseLayer(neurons=2, activation='relu'),
        DenseLayer(neurons=2, activation='sigmoid'),
        DenseLayer(neurons=1, activation='linear')
    ]

    model = SequentialModel(input_shape=(2,), layers=layers)

    # Create the training data for XOR
    X = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]

    # Corresponding target values
    y = [
        0.0,
        1.0,
        1.0,
        0.0
    ]

    X = eth.tensor.Tensor(data=X, shape=(4, 2))

    y = eth.tensor.Tensor(data=y, shape=(4,))

    model.train(X, y, epochs=10, learning_rate=0.001, verbose=True)

    # Evaluate the model
    predictions = model.predict(X)
    print("Final Predictions after training:")
    for i in range(len(predictions)):
        print(f"Input: {X[i].to_list()}, Predicted: {predictions[i].to_list()[0]}, Target: {y[i]}")

def main():

    # test_cuda()
    test_tensor_creation()
    # test_tensor_sum()
    # test_tensor_outer_product()
    # test_tensor_population()
    # test_tensor_dot_product()
    # test_model_creation()
    # test_activation_functions()
    # test_model_prediction()
    # test_model_training()

if __name__ == "__main__":
    main()
