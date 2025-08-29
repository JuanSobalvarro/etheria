from neuralscratchpy import ActivationFunctionType, MatrixNetwork, data_seq_normalization


def celsius2fahrenheit(celsius):
    return celsius * 9/5 + 32

def main():
    # Create a simple matrix network
    layer_sizes = [1, 1]
    hidden_activation = ActivationFunctionType.LINEAR
    output_activation = ActivationFunctionType.LINEAR
    model = MatrixNetwork(layer_sizes, hidden_activation, output_activation)

    input_data = []
    label_data = []
    for i in range(-300, 300, 1):
        input_data.append([i])
        label_data.append([celsius2fahrenheit(i)])

    # input_data = data_seq_normalization(input_data)
    # label_data = data_seq_normalization(label_data)

    model.train(input_data, label_data, epochs=10000, learning_rate=0.00001, verbose=True)

    output = model.predict([100])
    print("Output for [100]:", output)
    print("Expected output for [100]:", celsius2fahrenheit(100))

if __name__ == "__main__":
    main()