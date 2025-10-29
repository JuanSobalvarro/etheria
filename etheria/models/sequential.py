from typing import Tuple, List, Union, Sequence
from etheria.models.layers import Layer
from etheria.models.base import BaseModel
from etheria.models.types import ModelType
from etheria.activation import ActivationFunction, do_init, apply_activation, apply_derivative
from etheria.tensor import Tensor


class SequentialModel(BaseModel):
    """
    Sequential model that stacks layers linearly.
    """
    def __init__(self, input_shape: Tuple[int, ...], layers: Sequence[Layer]):
        super().__init__(input_shape, layers, model_type=ModelType.SEQUENTIAL)

        # each layer has a tensor of weights and biases
        # why list instead of a single tensor? because each layer can have different number of neurons so we need separate tensors
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
                # Input layer weights are the number of inputs
                input_size = self.flattened_input_size
            else:
                input_size = self.layers[i - 1].neurons
            
            # add weights for each neuron in the layer
            self.weights.append(Tensor(shape=(layer.neurons, input_size)))
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


    def train(self, X: Tensor, y: Tensor, epochs: int, verbose: bool = False, learning_rate: float = 0.01):
        """
        Train the sequential model using the provided training data (X, y) for a specified number of epochs.
        Currently supports basic SGD without mini-batching but when scaled it should select an optimizer.
        """

        # epoch loop
        for epoch in range(epochs):
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}: ")
            
            # forward pass
            predictions = self.predict(X)

            print(f"Predictions: {predictions}")

            # compute loss
            loss = self._compute_loss(y, predictions)
            if verbose:
                print(f"  Loss: {loss}")

            # compute gradients
            gradients = self._compute_gradients(X, y, loss)

            # backward pass (backpropagation)
            self._backpropagate(gradients, learning_rate)


    def predict(self, X: Tensor) -> Tensor:
        """
        This will perform a forward pass through the network and 
        return the output tensor which is a rank-2 tensor for n_batch_size x n_output_neurons.
        """

        # Ensure input is rank-2: (batch_size, input_dim)
        if len(X.shape) != 2:
            X = X.reshape((X.shape[0], self.flattened_input_size))

        # print(X)
        
        outputs: List[Tensor] = []
        
        for i in range(X.shape[0]):
            input_sample = X[i].reshape((self.flattened_input_size, 1))  # column vector
            
            # forward pass through each layer
            for layer_idx, layer in enumerate(self.layers):
                weights = self.weights[layer_idx]      # shape: (neurons, input_dim)
                biases = self.biases[layer_idx].reshape((layer.neurons, 1))  # column vector
                
                Z = weights.dot_product(input_sample) + biases
                input_sample = apply_activation(Z, layer.activation)
            
            outputs.append(input_sample)
        
        return outputs


    def evaluate(self, X, y, stats: List[str]) -> dict:
        ...

    def detail(self) -> str:
        details = f"Sequential Model:\n"
        details += f"Input Shape: {self.input_shape}\n"
        details += f"Layers:\n"
        for i, layer in enumerate(self.layers):
            details += f"  Layer {i + 1}:\n"
            details += f"    Layer Weights shape: {self.weights[i].shape}\n"
            for n in range(layer.neurons):
                neuron_weights = [self.weights[i][n, j] for j in range(self.weights[i].shape[1])]
                details += f"    Neuron {n + 1} Weights: {neuron_weights}, Bias: {self.biases[i][n]}, Activation: {layer.activation.name}\n"
        return details
    
    def _compute_loss(self, y: Tensor, predictions: Tensor) -> Tensor:
        """
        Compute Mean Squared Error loss between true labels and predictions. 
        Then it returns a Tensor rank-2 of shape (batch_size, output_neurons) representing the gradient of the loss w.r.t. predictions.
        """
        batch_size = len(predictions)
        total_loss = 0.0

        for i in range(batch_size):
            pred = predictions[i]  # column vector
            true = y[i].reshape(pred.shape)  # make sure same shape

            # MSE: (1/n) * sum((pred - true)^2)
            diff = pred - true
            squared_diff = diff * diff  # element-wise square
            sample_loss = 0.0
            for j in range(squared_diff.shape[0]):
                sample_loss = squared_diff[j, 0] + sample_loss  # sum over all outputs
            
            total_loss = sample_loss + total_loss

        mean_loss = total_loss * (1 / batch_size)
        return mean_loss

    def _compute_gradients(self, X: Tensor, y: Tensor, delta: Tensor) -> List[Tuple[Tensor, Tensor]]:
        """
        Returns a list of (dW, db) tuples for each layer.
        Where dW is the differential of the weights and db is the differential of the biases.
        So each layer i will have gradients[i] = (dW_i n_neurons x i-1_n_neurons, db_i n_neurons x 1)

        Also we have delta which is a rank-2 tensor of shape (batch_size, output_neurons)
        """
        grads: List[Tuple[Tensor, Tensor]] = []

        # Initialize gradients with zeros
        for layer in self.layers:
            dW = Tensor(shape=(layer.neurons, self.flattened_input_size))
            db = Tensor(shape=(layer.neurons, 1))
            grads.append((dW, db))

        # print(f"Initialized grads: {grads}")

        print(f"Delta shape: {delta}")
        
        # For each sample, accumulate gradients
        for i in range(delta.shape[0]):

            # print(f"Initial Delta: {delta}")

            # Backpropagate through layers
            for layer_idx in reversed(range(len(self.layers))):
                layer = self.layers[layer_idx]
                # so to calculate the gradient of hidden layers is the output of the previous layer * weights^T * delta * activation'
                A_prev: Tensor = Tensor(data=None, shape=(self.layers[layer_idx - 1].neurons, 1)) if layer_idx > 0 else X[i].reshape((self.flattened_input_size, 1))
                A_prev = apply_activation(A_prev, layer.activation) if layer_idx > 0 else A_prev
                # print(f"Layer {layer_idx}: A_prev shape: {A_prev.shape}, delta shape: {delta.shape}")

                # dL/dW = delta * A_prev^T
                dW = delta.dot_product(A_prev.transpose())
                
                # dL/db = delta
                db = delta # scalar

                # store gradients
                # print(f"Storing gradient {layer_idx}: dW shape {dW.shape}, db shape {db.shape}. Grad: {grads[layer_idx]}")
                grads[layer_idx] = (grads[layer_idx][0] + dW, grads[layer_idx][1] + db)
                
                # update delta for previous layer: delta_prev = W^T * delta * activation'
                if layer_idx > 0:
                    W = self.weights[layer_idx]
                    d1 = W.transpose().dot_product(delta)
                    d2 = apply_derivative(A_prev, self.layers[layer_idx-1].activation)
                    delta = d1 * d2  # element-wise multiplication

        # Average over batch
        for i in range(len(grads)):
            # print(f"Averaging gradient for layer {i} before: {grads[i]}")
            grads[i] = (grads[i][0] * (1 / batch_size), grads[i][1] * (1 / batch_size))
        
        return grads

    
    def _backpropagate(self, gradients: Sequence[Tuple[Tensor, Tensor]], learning_rate: float):
        """
        Once we have the gradients we update the weights and biases.
        """
        for i in range(len(self.layers)):
            dW, db = gradients[i]
            self.weights[i] = self.weights[i] - dW * learning_rate
            self.biases[i] = self.biases[i] - db.reshape(self.biases[i].shape) * learning_rate