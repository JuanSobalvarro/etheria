import os
import sys

# Locate the build directory dynamically
project_root = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(project_root, "build", "Release")

# Add build directory to PYTHONPATH temporarily
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

# Import the bindings
try:
    import neuralscratch
    print("Module loaded successfully:", neuralscratch)
except ModuleNotFoundError:
    print("Could not find the compiled module in:", build_dir)
    sys.exit(1)

# Test the functionality
nn = neuralscratch.NeuralNetwork(1, [], 1, neuralscratch.ActivationFunctionType.LINEAR)
nn.train([[1], [2], [3], [4], [5]], [[2], [4], [6], [8], [10]], 1000, 0.01)
nn.printNeuralNetwork()
nn.test([[1], [2], [3], [4], [5]], [[2], [4], [6], [8], [10]])
output = nn.predict([100])
print("Prediction for input 100:", output)
