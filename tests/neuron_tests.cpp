#include <iostream>
#include "neuron.hpp"


void test_1();
void test_2();
void test_3();

int main() {
    std::cout << "Running Neuron tests..." << std::endl;

    void (*tests[])() = {test_1, test_2, test_3};

    for (int i = 0; i < 3; ++i) {
        tests[i]();
    }

    return 0;
}

void test_1()
{
    Neuron neuron = Neuron(ActivationFunctionType::LINEAR);

    neuron.setWeights({1});
    neuron.setBias(0);
    Connection input;
    input.changeValue(1);
    neuron.setInputs({&input});
    double output = neuron.activate();

    std::cout << "Neuron output: " << output << std::endl;
    std::cout << "Neuron output (expected): " << 1 << std::endl;
}

void test_2()
{
    Neuron neuron = Neuron(ActivationFunctionType::SIGMOID);

    neuron.setWeights({1});
    neuron.setBias(0);
    Connection input;
    input.changeValue(1);
    neuron.setInputs({&input});
    double output = neuron.activate();

    std::cout << "Neuron output: " << output << std::endl;
    std::cout << "Neuron output (expected): " << 0.7310585786300049 << std::endl;
}

void test_3()
{
    Neuron neuron = Neuron(ActivationFunctionType::RELU);

    neuron.setWeights({1});
    neuron.setBias(0);
    Connection input;
    input.changeValue(-1);
    neuron.setInputs({&input});
    double output = neuron.activate();

    std::cout << "Neuron output: " << output << std::endl;
    std::cout << "Neuron output (expected): " << 0 << std::endl;
}