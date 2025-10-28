from typing import Tuple, List, Union, Sequence
from etheria.models.layers import Layer
from etheria.models.base import BaseModel
from etheria.models.types import ModelType
from etheria.activation import ActivationFunction, do_init
from etheria.tensor import Tensor


class SequentialModel(BaseModel):
    """
    Sequential model that stacks layers linearly.
    """
    def __init__(self, input_shape: Tuple[int, ...], layers: Sequence[Layer]):
        super().__init__(input_shape, layers, model_type=ModelType.SEQUENTIAL)

        # each layer has a tensor of weights and biases
        self.weights: List[Tensor] = []  
        self.biases: List[Tensor] = []

        self._initialize_weights_and_biases()

    def _initialize_weights_and_biases(self):
        """
        Initialize weights and biases for each layer in the sequential model. Based on the activation function used in each layer.
        """
        self.weights = []
        self.biases = []

        # go through each layer and create weights and biases tensors
        for i, layer in enumerate(self.layers):
            if i == 0:
                # Input layer
                self.weights.append(Tensor(shape=(layer.neurons, self.flattened_input_size)))
            else:
                self.weights.append(Tensor(shape=(layer.neurons, self.layers[i - 1].neurons)))

            # add biases for each neuron in the layer
            self.biases.append(Tensor(shape=(layer.neurons,)))

        # print(f"Weights shapes: {[w.shape for w in self.weights]}")
        # print(f"Biases shapes: {[b.shape for b in self.biases]}")

        # Now based on the activation function of each layer, we can initialize weights and biases
        for i, layer in enumerate(self.layers):
            if i == 0:
                # Input layer
                fan_in = self.flattened_input_size
            else:
                fan_in = self.layers[i - 1].neurons

            fan_out = layer.neurons
            # print(self.weights[i].shape, self.biases[i].shape, fan_in, fan_out, layer.activation)
            # we send weights matrix (tensor of shape (fan_out, fan_in)) and biases vector (tensor of shape (fan_out,))
            do_init(self.weights[i], self.biases[i], fan_in, fan_out, layer.activation)


    def train(self, X, y, epochs: int, verbose: bool = False):
        
        # convert images (X) and labels (y) to tensors
        X_tensor = Tensor(shape=(len(X),) + self.input_shape, data=X)
        y_tensor = Tensor(shape=(len(y), len(set(y))), data=y)  # assuming y is one-hot encoded

        # epoch loop
        for epoch in range(epochs):
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}: ")
            
            # forward pass


    def predict(self, X: Tensor) -> List[Tensor]:
        """
        Realizes a forward pass over a batch of inputs through the network to generate predictions.
        This can be done given an X input of shape (num_samples, input_shape...)
        Returns a Tensor of predictions.

        To do a forward pass algebraically, we need to do the following for each layer:
        For each layer l in layers:
            Z_l = W_l * A_(l-1) + b_l
            A_l = activation(Z_l)
        """
        batch_size = X.shape[0]
        inputs = X

        outputs = []
        for i, layer in enumerate(self.layers):
            Z = matmul(self.weights[i], inputs)
            Z = add(Z, self.biases[i])
            A = activation(Z, layer.activation.value)
            outputs.append(A)
            inputs = A

        return outputs

    def evaluate(self, X, y, stats: List[str]) -> dict:
        ...