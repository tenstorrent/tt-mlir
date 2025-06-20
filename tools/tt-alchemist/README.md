# tt-alchemist

A user-friendly abstraction layer between the tt-mlir compiler and end users. It simplifies the workflow for users by providing a unified interface for model conversion, building, and execution.

## Overview

tt-alchemist provides a simple command-line interface and a Python API for working with the tt-mlir compiler. It allows users to:

- Convert MLIR models to C++ code
- Generate complete, buildable solutions from the C++ code
- Build the solutions with different flavors (release, debug, profile)
- Run and profile the built solutions

## Installation

### Prerequisites

- tt-mlir environment activated
- CMake 3.16 or higher
- Python 3.6 or higher

### Building from Source

```bash
# Clone the repository (if not already done)
git clone https://github.com/tenstorrent/tt-mlir.git
cd tt-mlir

# Build tt-alchemist
mkdir -p build && cd build
cmake ..
make tt-alchemist

# Install the Python package
cd ../tools/tt-alchemist/python
pip install -e .
```

## Usage

### Command-Line Interface

```bash
# Convert a model to C++
tt-alchemist model-to-cpp model.mlir --output=my_model --opt-level=normal

# Build the generated solution
tt-alchemist build my_model --flavor=release --target=grayskull

# Run the built solution
tt-alchemist run my_model --input=input.bin --output=output.bin

# Profile the solution
tt-alchemist profile my_model --input=input.bin --report=profile.json

# List available hardware targets
tt-alchemist list-targets

# List available build flavors
tt-alchemist list-flavors
```

### Python API

```python
from tt_alchemist import TTAlchemist, OptimizationLevel, BuildFlavor, HardwareTarget

# Create a TTAlchemist instance
alchemist = TTAlchemist()

# Convert a model to C++
alchemist.model_to_cpp("model.mlir", {
    "opt_level": OptimizationLevel.NORMAL,
    "output_dir": "my_model"
})

# Build the generated solution
alchemist.build_solution("my_model", {
    "flavor": BuildFlavor.RELEASE,
    "target": HardwareTarget.GRAYSKULL
})

# Run the built solution
alchemist.run_solution("my_model", {
    "input_file": "input.bin",
    "output_file": "output.bin"
})

# Profile the solution
alchemist.profile_solution("my_model", {
    "input_file": "input.bin",
    "output_file": ""
}, "profile.json")
```

## Directory Structure

```
tt-alchemist/
├── include/                        # Public headers
│   └── tt-alchemist/
│       ├── tt_alchemist.h          # Main public API header
│       └── config.h                # Public configuration options
├── src/                            # Implementation files and private headers
│   ├── include/                    # Private headers
│   │   ├── compiler.h              # Compiler interface declarations
│   │   ├── solution_generator.h    # Solution generator declarations
│   │   └── runtime.h               # Runtime interface declarations
│   ├── compiler.cpp                # Compiler implementation
│   ├── solution_generator.cpp      # Solution generator implementation
│   ├── runtime.cpp                 # Runtime implementation
│   └── tt_alchemist.cpp            # Main API implementation
├── templates/                      # Templates for generated code
│   ├── cmake/                      # CMake templates
│   ├── run_scripts/                # Runtime script templates
│   └── project/                    # Project structure templates
├── python/                         # Python package and bindings
│   └── tt_alchemist/               # Python module
└── test/                           # Test code
```

## Generated Project Structure

When a solution is generated, it will have a structure like:

```
model_name/
├── CMakeLists.txt              # Main CMake file
├── README.md                   # Instructions for using the solution
├── src/
│   └── model_implementation.cpp # Generated C++ code
├── include/
│   └── model_interface.h       # Generated header
├── build_configs/
│   ├── release.cmake           # Release build configuration
│   ├── debug.cmake             # Debug build configuration
│   └── profile.cmake           # Profile build configuration
└── scripts/
    ├── build.sh                # Script to build the solution
    ├── run.sh                  # Script to run the solution
    └── profile.sh              # Script to profile the solution
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
