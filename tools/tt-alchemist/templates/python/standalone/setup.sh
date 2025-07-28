#!/bin/bash

# Setup script for tt-alchemist generated Python code using standalone ttnn wheel
# This script creates a virtual environment and installs the ttnn wheel

set -e

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

# Function to check if wheel needs rebuilding
should_rebuild_wheel() {
    local tt_metal_dir="$1/third_party/tt-metal/src/tt-metal"
    local dist_dir="$tt_metal_dir/dist"
    
    # If no wheels exist, need to build
    if [[ ! -d "$dist_dir" ]] || [[ $(ls "$dist_dir"/*.whl 2>/dev/null | wc -l) -eq 0 ]]; then
        return 0  # true - need to build
    fi
    
    # Get newest wheel timestamp
    local newest_wheel=$(ls -t "$dist_dir"/*.whl | head -1)
    local wheel_time=$(stat -c "%Y" "$newest_wheel")
    
    # Check if any source files are newer than the wheel
    local newer_files=$(find "$tt_metal_dir" -name "*.py" -o -name "*.cpp" -o -name "*.hpp" -o -name "CMakeLists.txt" | xargs stat -c "%Y" | sort -n | tail -1)
    
    if [[ $newer_files -gt $wheel_time ]]; then
        return 0  # true - need to rebuild
    fi
    
    return 1  # false - no need to rebuild
}

echo "=== tt-alchemist Standalone Python Setup ==="

# Find the tt-mlir repository root
TT_MLIR_ROOT=$(find_tt_mlir_root)
if [[ -z "$TT_MLIR_ROOT" ]]; then
    echo "Error: Could not find tt-mlir repository root"
    echo "Make sure you are running this script from within the tt-mlir repository"
    exit 1
fi

echo "Found tt-mlir repository at: $TT_MLIR_ROOT"

TT_METAL_DIR="$TT_MLIR_ROOT/third_party/tt-metal/src/tt-metal"
DIST_DIR="$TT_METAL_DIR/dist"

# Check if we need to build or rebuild the wheel
if should_rebuild_wheel "$TT_MLIR_ROOT"; then
    echo "Building ttnn wheel..."
    cd "$TT_METAL_DIR"
    
    # Activate the main env if available for building
    if [[ -f "$TT_MLIR_ROOT/env/bin/activate" ]]; then
        source "$TT_MLIR_ROOT/env/bin/activate"
    elif [[ -f "$TT_MLIR_ROOT/env/activate" ]]; then
        source "$TT_MLIR_ROOT/env/activate"
    fi
    
    python -m build --wheel
    echo "✓ Wheel built successfully"
else
    echo "✓ Wheel is up to date, skipping build"
fi

# Get the newest wheel
TTNN_WHEEL=$(ls -t "$DIST_DIR"/*.whl | head -1)
if [[ -z "$TTNN_WHEEL" ]]; then
    echo "Error: No ttnn wheel found in $DIST_DIR"
    exit 1
fi

echo "Using wheel: $(basename "$TTNN_WHEEL")"

# Go back to the template directory
cd "$(dirname "$0")"

# Create virtual environment if it doesn't exist
if [[ ! -d "venv" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install or upgrade the ttnn wheel
echo "Installing ttnn wheel..."
pip install --force-reinstall "$TTNN_WHEEL"

# Verify installation
echo "Verifying ttnn installation..."
if python3 -c "import ttnn" 2>/dev/null; then
    echo "✓ ttnn installed and imported successfully"
    python3 -c "import ttnn; print(f'ttnn version: {ttnn.__version__ if hasattr(ttnn, \"__version__\") else \"unknown\"}')"
else
    echo "✗ Failed to import ttnn after installation"
    exit 1
fi

echo ""
echo "✓ Setup complete!"
echo "Virtual environment is activated and ttnn is installed."
echo "You can now run: ./run"

