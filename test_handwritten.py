import numpy as np
import struct
import os
from neuralscratchpy import NeuralNetwork, ActivationFunctionType

def one_hot_encode_labels(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """One-hot encode a 1D NumPy array of labels."""
    encoded_labels = np.zeros((labels.size, num_classes), dtype=np.float64)
    encoded_labels[np.arange(labels.size), labels] = 1
    return encoded_labels

def load_idx_images_np(filename: str, normalize: bool = True, flatten: bool = True) -> np.ndarray:
    """Load MNIST image file into a NumPy array."""
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f"{filename} is not a valid MNIST image file (magic={magic})")
        
        data = np.frombuffer(f.read(), dtype=np.uint8).astype(np.float64)
        if normalize:
            data /= 255.0
        if flatten:
            data = data.reshape(num_images, rows * cols)
        else:
            data = data.reshape(num_images, rows, cols)
        return data

def load_idx_labels_np(filename: str) -> np.ndarray:
    """Load MNIST label file into a NumPy array."""
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f"{filename} is not a valid MNIST label file (magic={magic})")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def sample_per_class(images, labels, num_per_class=100):
    """Randomly sample a fixed number of examples per class."""
    sampled_images = []
    sampled_labels = []

    num_classes = labels.shape[1]  # one-hot labels
    for i in range(num_classes):
        idxs = np.where(labels[:, i] == 1)[0]
        selected = np.random.choice(idxs, min(num_per_class, len(idxs)), replace=False)
        sampled_images.append(images[selected])
        sampled_labels.append(labels[selected])

    return np.vstack(sampled_images), np.vstack(sampled_labels)

def main():
    data_dir = "./data"

    # Load full MNIST dataset
    train_images = load_idx_images_np(os.path.join(data_dir, "train-images.idx3-ubyte"))
    train_labels = load_idx_labels_np(os.path.join(data_dir, "train-labels.idx1-ubyte"))
    test_images = load_idx_images_np(os.path.join(data_dir, "t10k-images.idx3-ubyte"))
    test_labels = load_idx_labels_np(os.path.join(data_dir, "t10k-labels.idx1-ubyte"))

    # One-hot encode labels
    train_labels_encoded = one_hot_encode_labels(train_labels, 10)
    test_labels_encoded = one_hot_encode_labels(test_labels, 10)

    # Sample a smaller balanced training set
    train_images_small, train_labels_small = sample_per_class(train_images, train_labels_encoded, num_per_class=100)

    print("Training set:", train_images_small.shape, train_labels_small.shape)
    print("Test set:", test_images.shape, test_labels_encoded.shape)

    neural_network = NeuralNetwork(784, [128, 64], 10, ActivationFunctionType.RELU)

    try:
        neural_network.train(train_images_small, train_labels_small, 20, 0.01)
    except Exception as e:
        with open("error.log", "a") as f:
            f.write(f"Error during training: {e}\n")
        return

    # print for 3 first test
    # for i in range(3):
    #     test_prediction = neural_network.predict(test_images[i])
    #     test_prediction_number = np.argmax(test_prediction)
    #     expected_label = np.argmax(test_labels_encoded[i])
    #     print("Test prediction:", test_prediction)
    #     print("Expected:", expected_label)
    #     print("Predicted number:", test_prediction_number)

    neural_network.test(test_images, test_labels_encoded)

if __name__ == "__main__":
    main()
