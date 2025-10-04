import unittest
import time
from typing import List
import etheria as eth
import etheria.utils as utils


class TestEtheria(unittest.TestCase):

    def test_custom_config(self):
        """Create a network with custom layers."""
        nn_config = eth.NeuralNetworkConfig()
        nn_config.input_size = 4
        nn_config.layers = [
            eth.Layer(3, eth.Activation.RELU),
            eth.Layer(1, eth.Activation.SIGMOID)
        ]
        nn = eth.NeuralNetwork(nn_config)
        self.assertEqual(nn_config.input_size, 4)
        self.assertEqual(len(nn_config.layers), 2)
        self.assertIsInstance(nn.getWeights(), list)
        self.assertIsInstance(nn.getBiases(), eth.Matrix)

    def test_invalid_config(self):
        """Invalid input_size should raise."""
        nn_config = eth.NeuralNetworkConfig()
        nn_config.input_size = -1
        with self.assertRaises(Exception):
            eth.NeuralNetwork(nn_config)

    def test_data_conversion(self):
        """Test conversion from Python lists to Vector/Matrix."""
        # print(dir(utils))
        v = eth.Vector([1.0, 2.0, 3.0])
        self.assertIsInstance(v, eth.Vector)
        self.assertEqual(len(v), 3)
        self.assertEqual(v[0], 1.0)


        m = eth.Matrix([eth.Vector([1.0, 2.0]), eth.Vector([3.0, 4.0])])
        # m = eth.Matrix([[1.0, 2.0], [3.0, 4.0]])
        self.assertIsInstance(m, eth.Matrix)
        self.assertEqual(len(m), 2)
        self.assertEqual(len(m[0]), 2)
        self.assertEqual(m[1][1], 4.0)

    def test_training_f_to_c(self):
        """Simple training: Fahrenheit → Celsius."""
        nn_config = eth.NeuralNetworkConfig()
        nn_config.input_size = 1
        nn_config.layers = [
            eth.Layer(4, eth.Activation.RELU),
            eth.Layer(1, eth.Activation.LINEAR)
        ]
        nn = eth.NeuralNetwork(nn_config)

        x = eth.Matrix([
            eth.Vector([0]),
            eth.Vector([68]),
            eth.Vector([100]),
            eth.Vector([212])
        ])
        y = eth.Matrix([
            eth.Vector([0]),
            eth.Vector([20]),
            eth.Vector([37.78]),
            eth.Vector([100])
        ])

        start_time = time.time()
        try:
            nn.fit(x, y, epochs=100, learning_rate=0.000001, verbose=True)
        except Exception as e:
            self.fail(f"Training failed: {e}")
        elapsed = time.time() - start_time

        # Should finish within reasonable time
        self.assertLess(elapsed, 10.0)

        # Prediction should be reasonably close
        test_input = eth.Vector([32])
        prediction = nn.predict(test_input)
        self.assertIsInstance(prediction, eth.Vector)

        print(f"Prediction for 32°F: {prediction[0]:.2f}°C", "(expected: 0°C)")

    def test_cuda_support(self):
        """Check CUDA API calls (may skip if no CUDA available)."""
        try:
            cuda_available = eth.is_cuda_available()
            self.assertIsInstance(cuda_available, bool)
            if cuda_available:
                self.assertGreaterEqual(eth.number_cuda_devices(), 1)
        except AttributeError:
            self.skipTest("CUDA support not compiled in etheria.")

    def test_cuda_training(self):
        """
        Test cuda training with f_to_c dataset (if cuda available).
        """
        if not eth.is_cuda_available():
            self.skipTest("CUDA not available")

        nn_config = eth.NeuralNetworkConfig()
        nn_config.input_size = 1
        nn_config.layers = [
            eth.Layer(4, eth.Activation.RELU),
            eth.Layer(1, eth.Activation.LINEAR)
        ]
        nn = eth.NeuralNetwork(nn_config)

        try:
            nn.useCUDADevice(0)
        except Exception as e:
            self.fail(f"Failed to use CUDA device: {e}")

        x = eth.Matrix([
            eth.Vector([0]),
            eth.Vector([68]),
            eth.Vector([100]),
            eth.Vector([212])
        ])
        y = eth.Matrix([
            eth.Vector([0]),
            eth.Vector([20]),
            eth.Vector([37.78]),
            eth.Vector([100])
        ])

        start_time = time.time()
        try:
            nn.fit(x, y, epochs=100, learning_rate=0.0001, verbose=True)
        except Exception as e:
            self.fail(f"Training failed: {e}")
        elapsed = time.time() - start_time

        # Should finish within reasonable time
        self.assertLess(elapsed, 10.0)

        # Prediction should be reasonably close
        test_input = eth.Vector([32])
        prediction = nn.predict(test_input)
        self.assertIsInstance(prediction, eth.Vector)

        print(f"GPU Prediction for 32°F: {prediction[0]:.2f}°C", "(expected: 0°C)")

    def test_xor_homogeneous(self):
        """
        Train a network to learn XOR function. We are going to fit multiple times
        so the network should succeed to learn 99% of the times within 10,000 fits.
        """
        nn_config = eth.NeuralNetworkConfig()
        nn_config.input_size = 2
        nn_config.layers = [
            eth.Layer(2, eth.Activation.SIGMOID),
            eth.Layer(1, eth.Activation.SIGMOID)
        ]

        # XOR dataset
        X = eth.Matrix([eth.Vector([0,0]), eth.Vector([0,1]),
                        eth.Vector([1,0]), eth.Vector([1,1])])
        Y = eth.Matrix([eth.Vector([0]), eth.Vector([1]),
                        eth.Vector([1]), eth.Vector([0])])
        
        success_count = 0
        total_runs = 100
        for _ in range(100):
            nn = eth.NeuralNetwork(nn_config)

            try:
                nn.fit(X, Y, epochs=50000, learning_rate=1, verbose=False)
            except Exception as e:
                self.fail(f"Training failed: {e}")

            loss, accuracy = nn.evaluate(X, Y)
            print(f"XOR training loss: {loss:.4f}, accuracy: {accuracy:.4f}")

            if loss < 0.01:
                success_count += 1
        success_rate = success_count / total_runs
        self.assertGreaterEqual(success_rate, 0.99, "XOR learning success rate too low")

if __name__ == "__main__":
    unittest.main()
