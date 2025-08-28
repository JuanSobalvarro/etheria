import neuralscratchpy as nn
import random 

def celsius2fahrenheit(celsius: float) -> float:
    return (celsius * 9/5) + 32

def main():
    neural_network = nn.NeuralNetwork(1, [], 1, nn.ActivationFunctionType.LINEAR)

    # generate data and labels then do a 70/30 split
    data = []
    labels = []

    for _ in range(1000):
        celsius = random.uniform(-300, 300)
        fahrenheit = celsius2fahrenheit(celsius)
        data.append([celsius])
        labels.append([fahrenheit])

    # Split the data into training and testing sets (70/30 split)
    train_size = int(0.7 * len(data))
    train_data = data[:train_size]
    train_labels = labels[:train_size]
    test_data = data[train_size:]
    test_labels = labels[train_size:]

    # Train the neural network
    neural_network.train(training_data=train_data, labels=train_labels, epochs=5000, learning_rate=0.00001)

    neural_network.printNeuralNetwork()

    prediction = neural_network.predict([100])
    print(f"Predicted Fahrenheit for 100 Celsius: {prediction[0]}")

    # Test the neural network
    neural_network.test(test_data, test_labels)

if __name__ == "__main__":
    main()