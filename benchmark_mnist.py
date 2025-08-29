import time
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.datasets import mnist

import torch
import torch.nn as nn
import torch.optim as optim

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

from neuralscratchpy import NeuralNetwork, ActivationFunctionType, data_seq_normalization

# load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28) / 255.0
x_test = x_test.reshape(-1, 28*28) / 255.0

# One-hot encode labels
y_train_oh = to_categorical(y_train, num_classes=10)
y_test_oh = to_categorical(y_test, num_classes=10)

# config
layers = [784, 128, 64, 10]  # Same for all networks
epochs = 5
lr = 0.01
batch_size = 64

def scratch_nn():
    mnist_train = data_seq_normalization(x_train.tolist())
    mnist_test = data_seq_normalization(x_test.tolist())

    custom_model = NeuralNetwork(
        layers,
        hidden_activation=ActivationFunctionType.RELU,
        output_activation=ActivationFunctionType.SIGMOID  # for MSE loss
    )

    start = time.time()
    custom_model.train(mnist_train, y_train_oh.tolist(), epochs=epochs, learning_rate=lr, verbose=False)
    custom_time = time.time() - start

    loss, accuracy_custom = custom_model.evaluate(mnist_test, y_test_oh.tolist())

    return loss, accuracy_custom, custom_time

# PyTorch model
class TorchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

def torch_nn():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train_t = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_oh, dtype=torch.float32).to(device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test_oh, dtype=torch.float32).to(device)

    torch_model = TorchNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(torch_model.parameters(), lr=lr)

    start = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = torch_model(x_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
    torch_time = time.time() - start

    with torch.no_grad():
        outputs = torch_model(x_test_t)
        predicted = torch.argmax(outputs, axis=1)
        accuracy_torch = (predicted.cpu().numpy() == y_test).mean()

    return loss, accuracy_torch, torch_time


def tensor_nn():
    tf_model = Sequential([
        Dense(128, activation='relu', input_shape=(784,)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    tf_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    start = time.time()
    tf_model.fit(x_train, y_train_oh, epochs=epochs, batch_size=batch_size, verbose=0)
    tf_time = time.time() - start

    loss, accuracy_tf = tf_model.evaluate(x_test, y_test_oh, verbose=0)\
    
    return loss, accuracy_tf, tf_time


def main():

    loss_custom, accuracy_custom, custom_time = scratch_nn()
    loss_torch, accuracy_torch, torch_time = torch_nn()
    loss_tf, accuracy_tf, tf_time = tensor_nn()

    print("Training time (s):")
    print(f"PyTorch: {torch_time:.2f}, TensorFlow: {tf_time:.2f}, Custom: {custom_time:.2f}")

    print("\nTest accuracy:")
    print(f"PyTorch: {accuracy_torch*100:.2f}%, TensorFlow: {accuracy_tf*100:.2f}%, Custom: {accuracy_custom*100:.2f}%")


if __name__ == "__main__":
    main()

