# `tt-alchemist`

`tt-alchemist` is a code generation tool that converts MLIR models to executable C++ or Python solutions for Tenstorrent AI accelerators.

## Table of Contents
- [Support Matrix](#support-matrix)
- [Usage](#usage)
  - [Using via CLI](#using-via-cli)
  - [Usage via lib](#usage-via-lib)


## Support Matrix

> _Note: The tool is currently in development and is subject to frequent changes. Please refer to this document for most up-to-date information. Support matrix is provided below._

The following table summarizes the current support for code generation modes in `tt-alchemist`:

|                | C++                     | Python |
|----------------|-------------------------|--------------|
| **standalone** | ✅ Supported            | ❌ Not yet supported |
| **local**      | 🟨 Experimental support | 🟨 Experimental support |

Modes:
- **standalone**: Generates a self-contained solution with all necessary dependencies copied into the output directory. Useful for deployment and sharing.
- **local**: Generates code that uses libraries from the source tree, minimizing duplication and disk usage. Useful for development and debugging.

> _Note: Python codegen currently supports a small subset of operations compared to C++. Full support is being actively worked on and is coming soon._

## Usage

The tool is compiled into a C++ library, with a thin CLI wrapper written in Python. This means that it can be distributed both as a C++ library, and as a CLI tool via Python wheel mechanism.

### Using via CLI

To use via CLI, it is suggested to build the tool from source. Alternatively, look for `tt-alchemist` artifacts within [CI runs](https://github.com/tenstorrent/tt-mlir/actions/workflows/on-push.yml).

```bash
# Assuming the user had already built the tt-mlir compiler and turned on the python virtual env

# Build the tt-alchemist lib, package into Python wheel, and install to active env
cmake --build build -- tt-alchemist
```

For all available CLI options and usage instructions, run:
```bash
tt-alchemist --help
```

All APIs today accept a `.mlir` file that describe a model in `TTIR` dialect.
Example usage:
```bash
# Generate a whole standalone C++ solution and run
tt-alchemist generate-cpp tools/tt-alchemist/test/models/mnist.mlir -o mnist_cpp --standalone
cd mnist_cpp
./run

# Similar to above, but use "local" libs from source dir - this saves on memory by not copying the whole dev package to the output dir
tt-alchemist generate-cpp tools/tt-alchemist/test/models/mnist.mlir -o mnist_cpp --local
cd mnist_cpp
./run

# Similarly for python
tt-alchemist generate-python tools/tt-alchemist/test/models/mnist.mlir -o mnist_python --local
cd mnist_python
./run

# Following APIs are intended to be used for debugging purposes

# Convert a mlir file to C++ code and print to console
tt-alchemist model-to-cpp tools/tt-alchemist/test/models/mnist.mlir

# Same, but for python (current support limited to few ops)
tt-alchemist model-to-python tools/tt-alchemist/test/models/mnist.mlir
```

### Usage via lib

To use within another project (e.g. a frontend like `tt-xla`), build the library from source:
```bash
# Assuming the user had already built the tt-mlir compiler and turned on the python virtual env

# Build the tt-alchemist lib
cmake --build build -- tt-alchemist-lib
```

Then, you may call any of the APIs listed [here](/tools/tt-alchemist/include/tt-alchemist/tt_alchemist_c_api.hpp).
