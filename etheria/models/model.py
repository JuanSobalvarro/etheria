from typing import Tuple, List
from etheria.models.layers import Layer
from etheria.dtypes import DType
from enum import Enum


class ModelType(Enum):
    BASE = "base" # technically we wont use this value but it is here for completeness
    SEQUENTIAL = "sequential"


class BaseModel:
    """
    Base class for machine learning models.
    """
    def __init__(self, input_shape: Tuple[int, ...], layers: List[Layer], dtype: DType, model_type: ModelType = ModelType.BASE):
        self.input_shape: Tuple[int, ...] = input_shape
        self.layers: List[Layer] = layers
        self.model_type: ModelType = model_type
        self.dtype: DType = dtype

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
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, verbose: bool = False):
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def evaluate(self, X: np.ndarray, y: np.ndarray, stats: List[str]) -> dict:
        ...        


class SequentialModel(BaseModel):
    """
    Sequential model that stacks layers linearly.
    """
    def __init__(self, input_shape: Tuple[int, ...], layers: List[Layer], dtype: DType):
        super().__init__(input_shape, layers, dtype=dtype, model_type=ModelType.SEQUENTIAL)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, verbose: bool = False):
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, stats: List[str]) -> dict:
        ...