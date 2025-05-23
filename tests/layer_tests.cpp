#include <iostream>
#include "layer.hpp"


void test_1();
void test_2();
void test_3();

int main()
{
    std::cout << "Running layers tests..." << std::endl;

    void (*tests[])() = {test_1, test_2, test_3};

    for (int i = 0; i < 3; ++i) {
        tests[i]();
    }

    return 0;
}

void test_1()
{
    std::cout << "Running test 1..." << std::endl;

    // input layer
    std::vector<Connection*> inputs = {new Connection(), new Connection()};

    Layer layer(2, inputs, ActivationFunctionType::LINEAR);
    layer.setWeights({{1, 1}, {1, 1}});
    layer.setBiases({0, 0});

    inputs[0]->changeValue(1);
    inputs[1]->changeValue(1);
    std::vector<double> output = layer.activate();
    std::cout << "Layer output: ";
    for (double val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    std::cout << "Layer output (expected): 2 2" << std::endl;
    std::cout << "Test 1 completed." << std::endl;
}

void test_2()
{

}

void test_3()
{

}