
# PYTHON LIB DIRECTORY

PYTHON_LIB_DIR := './etheria/core/'
PYTHON_LIB_NAME := 'etheria.cp313-win_amd64.pyd'

@_default:
    just --list

[group('build')]
@build_copy: build copy

# build C++ test executable
[group('build')]
@build:
    uv run cmake -S . -B build
    uv run cmake --build build --config Release

[group('build')]
@build_debug:
    uv run cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
    uv run cmake --build build --config Debug

[group('build')]
@copy:
    cp -r build/Release/etheria.* {{ PYTHON_LIB_DIR }}
    echo "Copied libraries to {{ PYTHON_LIB_DIR }}"

[group('clean')]
@clean:
    rm -rf build

[group('cpptests')]
@run_cpp_tests:
    uv run build/cpptests/Debug/test_fit.exe