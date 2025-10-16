import etheria as eth

def test_cuda():
    print(f"The number of CUDA devices is: {eth.cuda.number_cuda_devices()}")
    print(f"The current CUDA device is: {eth.cuda.current_cuda_device()}")
    print(f"The details of CUDA device 0 are: {eth.cuda.device_details(0)}")

def test_tensor():
    a = eth.tensor.Tensor(shape=[3, 4])

    # set each item i, j from 1 to 12
    for i in range(3):
        for j in range(4):
            a[i][j] = i * 4 + j + 1
    print(a[0])
    print(a)
    # test put single value
    a[0][0] = 99
    a[2][3] = 100
    print("Tensor data after setting a[0][0] = 99 and a[2][3] = 100:", a)

def test_tensor_ops():
    a = eth.tensor.Tensor(shape=[2, 3])
    b = eth.tensor.Tensor(shape=[3, 2])

    # set values for a
    for i in range(2):
        for j in range(3):
            a[i][j] = i * 3 + j + 1  # 1 to 6

    # set values for b
    for i in range(3):
        for j in range(2):
            b[i][j] = i * 2 + j + 1  # 1 to 6

    print("Tensor a:", a)
    print("Tensor b:", b)

    c = eth.tensor.add(a, a)
    print("Tensor c (a + a):", c)
    print(f"Expected c data: {[2, 4, 6, 8, 10, 12]}")

    d = eth.tensor.matmul(a, b)
    print("Tensor d (a @ b):", d)
    print(f"Expected d data: {[22, 28, 49, 64]}")

    e = eth.tensor.activation(a, 'relu')
    print("Tensor e (ReLU activation):", e)
    print(f"Expected e data: {[1, 2, 3, 4, 5, 6]}")
    e = eth.tensor.activation(a, 'sigmoid')
    print("Tensor e (Sigmoid activation):", e)
    print(f"Expected e data: {[1/(1+2.718281828459045**-x) for x in [1,2,3,4,5,6]]}")

def test_model():
    model = eth.models.SequentialModel(input_shape=(2,), layers=[
        eth.models.DenseLayer(neurons=4, activation='relu'),
        eth.models.DenseLayer(neurons=1, activation='tanh')
    ])

    model.summary()

    # print(model.weights)

    # try predict
    pred_inp = [[0.5, -1.5], [1.0, 2.0]]
    predictions = model.predict(pred_inp)
    for i, pred in enumerate(predictions):
        print(f"Prediction {i} for {pred_inp[i]}: {pred}")


def main():

    test_cuda()
    # test_tensor()
    # test_tensor_ops()
    # test_model()

if __name__ == "__main__":
    main()