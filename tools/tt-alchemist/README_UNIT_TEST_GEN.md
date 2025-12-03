# TTNN Unit Test Generation from MLIR

## Overview

This feature extends the tt-alchemist tool to automatically generate parametrized Python unit tests from TTNN Dialect MLIR. It analyzes TTNN operations in MLIR modules and creates comprehensive pytest-based test suites.

## Features

- **Automatic Test Generation**: Converts TTNN MLIR operations into executable Python unit tests
- **Parametrization**: Groups similar operations and creates parametrized tests for efficient testing
- **Operation Filtering**: Generate tests for specific operations only
- **Complete Test Infrastructure**: Generates conftest.py, test utilities, and test files

## Architecture

### Components

1. **MLIRModuleSplitter** (C++)
   - Splits MLIR modules into individual operations
   - Handles DeviceModuleOp unwrapping
   - Processes operations in execution order

2. **TestParametrizer** (C++)
   - Groups similar operations for parametrization
   - Extracts unique parameter combinations
   - Generates test cases with proper IDs

3. **UnitTestGenerator** (C++)
   - Orchestrates the test generation process
   - Creates Python test files using templates
   - Integrates with EmitPy for code generation

4. **Python API**
   - Provides convenient Python interface
   - Supports both file and string input

## Usage

### Python API

```python
from tt_alchemist import generate_unit_tests

# Generate tests from MLIR file
generate_unit_tests(
    "model.mlir",
    "tests/",
    op_filter=["ttnn.add", "ttnn.relu"],
    parametrized=True,
    verbose=True
)
```

### Command Line (when CLI is implemented)

```bash
# Generate all tests
tt-alchemist gen-tests -i model.mlir -o tests/

# Generate tests for specific ops
tt-alchemist gen-tests -i model.mlir -o tests/ --ops ttnn.add,ttnn.relu

# Disable parametrization
tt-alchemist gen-tests -i model.mlir -o tests/ --no-parametrize
```

## Generated Test Structure

```
tests/
├── conftest.py              # Device fixtures and setup
├── test_utils.py            # Helper functions
├── test_ttnn_add.py         # Parametrized tests for add ops
├── test_ttnn_relu.py        # Parametrized tests for relu ops
└── test_ttnn_matmul.py      # Parametrized tests for matmul ops
```

### Example Generated Test

```python
import pytest
import torch
import ttnn
from test_utils import create_random_tensor, validate_output

class TestTtnnAdd:
    """Auto-generated tests for ttnn.add operation."""

    @pytest.mark.parametrize("shape", [
        (1, 32, 128, 128),
        (1, 64, 256, 256)
    ])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_add_parametrized(self, shape, dtype, device):
        """Test add with various parameters."""
        input0 = create_random_tensor(shape, dtype)
        input1 = create_random_tensor(shape, dtype)

        ttnn_input0 = ttnn.from_torch(input0, device=device)
        ttnn_input1 = ttnn.from_torch(input1, device=device)

        # Execute TTNN operation
        output = ttnn.add(ttnn_input0, ttnn_input1)

        # Compute golden values
        expected = torch.add(input0, input1)

        # Validate output
        actual = ttnn.to_torch(output)
        assert validate_output(actual, expected)
```

## Implementation Details

### Operation Grouping

Operations are grouped by:
- Operation name (e.g., all `ttnn.add` operations)
- Number of inputs/outputs
- Compatible attributes

### Parametrization Strategy

When multiple similar operations are found:
1. Extract unique parameter combinations (shapes, dtypes, attributes)
2. Generate parametrized test covering all variations
3. Ensure test case uniqueness with proper IDs

### Integration with EmitPy

The system can leverage the existing TTNNToEmitPy conversion pipeline to generate Python code for operations, though the current implementation generates tests directly using templates.

## Building

Add the following files to the CMakeLists.txt:
```cmake
set(LIB_SOURCES
  # ... existing sources ...
  lib/mlir_module_splitter.cpp
  lib/test_parametrizer.cpp
  lib/unit_test_generator.cpp
)
```

## Testing

Run the test script to verify the implementation:
```bash
python test/python/test_alchemist_unit_gen.py
```

## Future Enhancements

1. **Full EmitPy Integration**: Use EmitPy to generate operation code
2. **Attribute Support**: Better handling of operation attributes in tests
3. **Golden Value Generation**: Smart golden value computation based on op type
4. **Coverage Reporting**: Track which operations have tests
5. **Custom Templates**: Allow users to provide custom test templates
6. **Multiple Test Frameworks**: Support for unittest, nose, etc.

## Status

This is the initial implementation providing the core infrastructure for unit test generation. The following components are complete:

- ✅ C++ infrastructure (splitter, parametrizer, generator)
- ✅ Python API bindings
- ✅ Test templates and utilities
- ✅ CMake integration
- ⏳ CLI integration (pending)
- ⏳ Full C API marshaling (simplified version implemented)

## Notes

- The current Python API uses simplified placeholder implementation for C struct marshaling
- Full integration requires proper ctypes.Structure for TestGenerationOptions
- The CLI command needs to be added to the main tt-alchemist executable