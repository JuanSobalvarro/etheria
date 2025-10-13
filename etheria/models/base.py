from typing import Tuple, List, Sequence
from etheria.models.layers import Layer
from etheria.models.types import ModelType


class BaseModel:
    """
    Base class for machine learning models.
    """
    def __init__(self, input_shape: Tuple[int, ...], layers: Sequence[Layer], model_type: ModelType = ModelType.BASE):
        self.input_shape: Tuple[int, ...] = input_shape
        self.flattened_input_size: int = self.get_flattened_input_size()
        self.layers: Sequence[Layer] = layers
        self.model_type: ModelType = model_type

    def get_flattened_input_size(self) -> int:
        """
        Calculate the flattened size of the input shape.
        """
        size = 1
        for dim in self.input_shape:
            size *= dim
        return size

    def summary(self) -> str:
        """
        Returns a summary of the model architecture.
        """
        summary_str = f"Model Type: {self.model_type}\n"
        summary_str += f"Input Shape: {self.input_shape}\n"
        summary_str += "Layers:\n"
        for i, layer in enumerate(self.layers):
            summary_str += f"  Layer {i + 1}: {layer} with {layer.neurons} neurons and {layer.activation} activation\n"
        return summary_str
    
    def configure(self, learning_rate: float, optimizer: str, loss: str):
        """
        Configures the model with learning parameters.
        """
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss = loss
    
    def train(self, X, y, epochs: int, verbose: bool = False):
        ...

    def predict(self, X):
        ...

    def evaluate(self, X, y, stats: List[str]) -> dict:
        ...        