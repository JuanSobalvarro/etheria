from typing import Tuple, List, Union, Sequence
from etheria.models.layers import Layer, DenseLayer
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


    def train(self, inputs: Tensor, targets: Tensor, epochs: int, verbose: bool = False, learning_rate: float = 0.01):
        """
        Train the sequential model using the provided training data (X, y) for a specified number of epochs.
        Currently supports basic SGD without mini-batching but when scaled it should select an optimizer.
        """

        # epoch loop
        for epoch in range(epochs):
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}: ")
            
            # forward pass
            predictions = self.predict(inputs)

            # print(f"Predictions: {predictions}")

            # compute loss
            loss = self._compute_loss(targets, predictions)
            if verbose:
                print(f"  Loss: {loss}")

            # compute gradients
            gradients = self._compute_gradients(inputs, targets, loss)

            # backward pass (backpropagation)
            self._backpropagate(gradients, learning_rate)


    def predict(self, inputs: Tensor) -> Tensor:
        """
        This will perform a forward pass through the network and 
        return the output tensor which is a rank-2 tensor for n_batch_size x n_output_neurons.
        """

        # Ensure input is rank-2: (batch_size, input_dim)
        if inputs.rank != 2:
            raise ValueError(f"Input tensor X must be rank-2 (batch_size, input_dim). Got rank-{inputs.rank} with shape {inputs.shape}.")

        # print(X)
        
        # outputs: Tensor = Tensor(shape=(X.shape[0], self.layers[-1].neurons)) # output tensor of shape (batch_size, output_neurons)

        # temporaly set item only support floats, we need to fix that so we can set tensors directly, for now lets use a list of tensors then we will conver to a single tensor at the end
        outputs: List[Tensor] = []
        
        for i in range(inputs.shape[0]):
            input_sample = inputs[i].reshape((self.flattened_input_size,))  # column vector
            
            # forward pass through each layer
            for layer_idx, layer in enumerate(self.layers):
                weights = self.weights[layer_idx]     # shape: (neurons, input_dim)
                print(f"Weights shape: {weights.shape}")
                biases = self.biases[layer_idx].reshape((layer.neurons,))  # column vector

                layer.forward(input_sample, weights, biases)
            
                input_sample = layer.output  # output of current layer is input to next layer

            outputs.append(input_sample)

        output_data = [outputs[i].to_list() for i in range(len(outputs))]
        output = Tensor(data=output_data, shape=(inputs.shape[0], self.layers[-1].neurons))

        return output


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
    
    def _compute_loss(self, targets: Tensor, predictions: Tensor) -> Tensor:
        """
        Compute Mean Squared Error loss between true labels and predictions. 
        Then it returns a Tensor rank-2 of shape (batch_size, output_neurons) representing the gradient of the loss w.r.t. predictions.
        """
        batch_size = predictions.shape[0]
        total_loss = Tensor(data=[0.0]) # vector of shape (n_output_neurons,)

        # print(f"Total loss tensor: {total_loss}")

        for i in range(batch_size):
            pred_sample = predictions[i] # vector
            true_sample = targets[i]  # vector

            # print(f"Pred sample: {pred_sample}, True sample: {true_sample}")

            # MSE loss and its gradient
            sample_loss = (pred_sample - true_sample) * (pred_sample - true_sample)  # element-wise squared error
            sample_loss = sample_loss * 0.5  # to simplify derivative

            # print(f"Sample loss tensor: {sample_loss}")
            
            total_loss = sample_loss + total_loss

        mean_loss = total_loss * (1 / batch_size)

        return mean_loss

    def _compute_gradients(self, inputs: Tensor, outputs: Tensor, delta: Tensor) -> List[Tuple[Tensor, Tensor]]:
        """
        Returns a list of (dW, db) tuples for each layer.
        Where dW is the differential of the weights and db is the differential of the biases.
        So each layer i will have gradients[i] = (dW_i n_neurons x i-1_n_neurons, db_i n_neurons x 1)

        Also we have delta which is a rank-2 tensor of shape (batch_size, output_neurons)
        """
        grads: List[Tuple[Tensor, Tensor]] = []

        # Initialize gradients with zeros
        
        # For each sample, accumulate gradients
        for i in range(delta.shape[0]):
            current_delta = delta[i]  # shape: (output_neurons,)
            input_size = self.flattened_input_size

            # Backpropagate through layers
            for layer_idx in reversed(range(len(self.layers))):
                layer = self.layers[layer_idx]

                current_input = inputs[i] if layer_idx == 0 else self.layers[layer_idx - 1].output  # shape: (input_dim,)

                weights = self.weights[layer_idx]      # shape: (neurons, input_dim)
                biases = self.biases[layer_idx]

                dW_sample = current_delta.outer_product(current_input) # shape: (neurons, input_dim)
                db_sample = current_delta  # shape: (neurons,)

        
        return grads

    
    def _backpropagate(self, gradients: Sequence[Tuple[Tensor, Tensor]], learning_rate: float):
        """
        Once we have the gradients we update the weights and biases.
        """
        for i in range(len(self.layers)):
            dW, db = gradients[i]
            self.weights[i] = self.weights[i] - dW * learning_rate
            self.biases[i] = self.biases[i] - db.reshape(self.biases[i].shape) * learning_rate