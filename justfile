
# PYTHON LIB DIRECTORY

PYTHON_LIB_DIR := './etheria/'
PYTHON_LIB_NAME := 'etheria.cp313-win_amd64.pyd'

@_default:
    just --list

[group('build')]
@build_copy: build copy_libs

# build C++ test executable
[group('build')]
@build:
    uv run cmake -S . -B build
    uv run cmake --build build --config Release

[group('build')]
@copy_libs:
    cp -r build/Release/etheria.* {{ PYTHON_LIB_DIR }}
    echo "Copied libraries to {{ PYTHON_LIB_DIR }}"
