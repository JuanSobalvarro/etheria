import etheria as eth

def main():
    nn = eth.NeuralNetwork(layer_sizes=[1, 3, 1],
                           hidden_activation=eth.ActivationFunctionType.RELU,
                           output_activation=eth.ActivationFunctionType.LINEAR)
    print("Neural Network created successfully.")
    print("Weights:", nn.weights)
    print("Biases:", nn.biases)

"""
HOW I DESIRE TO USE IT:
layers = [
    eth.Layer(1, activation=eth.ActivationFunctionType.RELU), # input layer
    eth.Layer(3, activation=eth.ActivationFunctionType.RELU), # hidden layer
    eth.Layer(1, activation=eth.ActivationFunctionType.LINEAR), # output layer
]

nn = eth.NeuralNetwork(layers=layers, ...)

nn.train(...)

nn.predict(...)

"""
if __name__ == "__main__":
    main()