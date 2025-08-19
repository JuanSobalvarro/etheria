# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++23 -Wall -O2 -Iinclude -g

# Directories
SRC_DIR := src
TEST_DIR := tests
BUILD_DIR := build

# All source files in src/
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
SRC_OBJS  := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRC_FILES))

# Main program
TARGET := neural_network
MAIN_SRC := main.cpp
MAIN_OBJ := $(BUILD_DIR)/main.o

# Default build
all: $(TARGET)

$(TARGET): $(MAIN_OBJ) $(SRC_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Generic test rule
# Usage: make test TEST=nn_tests
test:
	@if [ -z "$(TEST)" ]; then \
		echo "Usage: make test TEST=<test_name>"; \
		exit 1; \
	fi
	$(MAKE) $(TEST)

# Example test target
nn_tests: $(BUILD_DIR)/nn_tests.o $(SRC_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compilation rules
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Ensure build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean
clean:
	rm -rf $(BUILD_DIR) $(TARGET) nn_tests

.PHONY: all clean test
