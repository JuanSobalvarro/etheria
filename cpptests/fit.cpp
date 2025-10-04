#include "mlp/neural_network.hpp"
#include <iostream>

using namespace eth::mlp;
using namespace eth::algebra;

int main() {
    try {
        NeuralNetworkConfig cfg;
        cfg.input_size = 1;
        cfg.layers = { Layer(1, eth::algebra::activation::LINEAR), Layer(1, eth::algebra::activation::LINEAR) };

        NeuralNetwork nn(cfg);

        std::vector<Vector> x = { Vector{32}, Vector{68}, Vector{100}, Vector{212} };
        std::vector<Vector> y = { Vector{0}, Vector{20}, Vector{37.78}, Vector{100} };

        nn.fit(x, y, 10, 0.01, true);

        std::cout << "C++ fit test finished successfully.\n";
    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
