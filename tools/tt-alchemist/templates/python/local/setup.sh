#!/bin/bash

# Setup script for tt-alchemist generated Python code using local ttnn library
# This script configures the environment to use the local checkout's ttnn python lib

# Function to find the tt-mlir repository root
find_tt_mlir_root() {
    local current_dir="$(pwd)"
    while [[ "$current_dir" != "/" ]]; do
        if [[ -f "$current_dir/CMakeLists.txt" ]] && [[ -d "$current_dir/third_party/tt-metal" ]]; then
            echo "$current_dir"
            return 0
        fi
        current_dir="$(dirname "$current_dir")"
    done
    return 1
}

# Find the tt-mlir repository root
TT_MLIR_ROOT=$(find_tt_mlir_root)
if [[ -z "$TT_MLIR_ROOT" ]]; then
    echo "Error: Could not find tt-mlir repository root"
    echo "Make sure you are running this script from within the tt-mlir repository"
    exit 1
fi

echo "Found tt-mlir repository at: $TT_MLIR_ROOT"

# Set up Python path to include the local ttnn library
export PYTHONPATH="${TT_MLIR_ROOT}/build/python_packages:${TT_MLIR_ROOT}/.local/toolchain/python_packages/mlir_core:${TT_MLIR_ROOT}/third_party/tt-metal/src/tt-metal/tt_eager:${TT_MLIR_ROOT}/third_party/tt-metal/src/tt-metal/build/tools/profiler/bin:${TT_MLIR_ROOT}/third_party/tt-metal/src/tt-metal/ttnn:${TT_MLIR_ROOT}/third_party/tt-metal/src/tt-metal/ttnn:${TT_MLIR_ROOT}/third_party/tt-metal/src/tt-metal:$PYTHONPATH"

# Activate the virtual environment if it exists
if [[ -f "$TT_MLIR_ROOT/env/bin/activate" ]]; then
    echo "Activating virtual environment..."
    source "$TT_MLIR_ROOT/env/bin/activate"
elif [[ -f "$TT_MLIR_ROOT/env/activate" ]]; then
    echo "Activating virtual environment..."
    source "$TT_MLIR_ROOT/env/activate"
else
    echo "Warning: No virtual environment found at $TT_MLIR_ROOT/env/"
    echo "You may need to activate the tt-mlir virtual environment manually"
fi

# Install ttnn dependencies if not already installed
echo "Installing ttnn dependencies..."
TTNN_DEPS="loguru>=0.6.0 numpy>=1.24.4,<2 networkx>=3.1 graphviz>=0.20.3"
if ! python3 -c "import loguru" 2>/dev/null; then
    echo "Installing missing ttnn dependencies..."
    python3 -m pip install $TTNN_DEPS
else
    echo "ttnn dependencies already installed"
fi

# Verify that ttnn can be imported
echo "Verifying ttnn import..."
if python3 -c "import ttnn" 2>/dev/null; then
    echo "✓ ttnn imported successfully"
    python3 -c "import ttnn; print(f'ttnn location: {ttnn.__file__}')"
else
    echo "✗ Failed to import ttnn"
    echo "Make sure the tt-mlir environment is properly set up"
    exit 1
fi

echo "Environment setup complete!"
echo "You can now run: ./run"
