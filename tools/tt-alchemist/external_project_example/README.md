# External Project Example

This example demonstrates how to use tt-alchemist from an external project (i.e. using ExternalProject_Add(tt-mlir ...))

## Building and Running

```bash
# From tt-mlir root directory
cd tools/tt-alchemist/external_project_example
cmake -B build -DTTMLIR_BUILD_DIR=$TT_MLIR_HOME/build
cmake --build build
./build/example_usage
```
