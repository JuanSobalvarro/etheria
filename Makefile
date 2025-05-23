CXX = g++
CXXFLAGS = -std=c++23 -Wall -O2 -Iinclude -g

TARGET = neural_network
SRCS = main.cpp src/neural_network.cpp
OBJS = $(notdir $(SRCS:.cpp=.o))

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Layer tests
LAYER_TEST_SRCS = tests/layer_tests.cpp src/layer.cpp src/neuron.cpp src/connection.cpp
LAYER_TEST_OBJS = $(notdir $(LAYER_TEST_SRCS:.cpp=.o))
LAYER_TEST_TARGET = layer_tests

$(LAYER_TEST_TARGET): $(LAYER_TEST_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: tests/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJS) $(TARGET) $(LAYER_TEST_OBJS) $(LAYER_TEST_TARGET)
.PHONY: all clean
