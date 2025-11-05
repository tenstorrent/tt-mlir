# External Project Example

This example demonstrates how to use tt-alchemist from an external project.

## Building and Running

```bash
# From this directory
cd tools/tt-alchemist/external_project_example
cmake -B build -DTTMLIR_BUILD_DIR=$TT_MLIR_HOME/build
cmake --build build
./build/example_usage
```

## Expected Output

```
Testing tt-alchemist from external project...
✅ Templates directory: /path/to/build/tools/tt-alchemist/templates
📁 Available templates:
  - cpp/
  - python/
✅ All tests passed!
```
