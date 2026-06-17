# D2M-JIT Pattern Testing Framework

This directory contains the refactored pattern testing infrastructure for d2m-jit.

## Overview

The pattern testing framework allows pattern developers to define test metadata directly in pattern definition files (`tools/d2m-jit/patterns/*.py`), which is then automatically discovered and executed as both LIT tests and on-device E2E tests.

## Architecture

### Pattern Definition Files

Pattern files in `tools/d2m-jit/patterns/` now contain:
1. The pattern implementation (kernel functions and pattern decorators)
2. Test metadata in a `PATTERN_TEST_METADATA` dictionary

Example structure:

```python
# tools/d2m-jit/patterns/eltwise_exp_to_kernel.py

@d2m.kernel
def exp_fused(in_t, out_t, m_blocks, n_blocks):
    # ... kernel implementation ...

@d2m.pattern(root=ttir.ExpOp, benefit=10)
def lower_exp(op, rewriter):
    # ... pattern implementation ...

# Test metadata
PATTERN_TEST_METADATA = {
    "pattern_name": "eltwise_exp",
    "description": "Single-eltwise pattern: ttir.exp -> d2m subgraph",

    "lit_tests": [
        {
            "name": "exp_pattern_positive",
            "module_text": "...",  # MLIR module to test
            "file_checks": ["CHECK: ...", ...],  # FileCheck patterns
        }
    ],

    "e2e_tests": [
        {
            "name": "test_pattern_exp_kernel_on_device",
            "kernel_fn": exp_fused,
            "input_generator": lambda: {"x": torch.rand(32, 32)},
            "reference_fn": lambda x: torch.exp(x),
            "layout_config": {...},
            "kernel_args": {...},
        }
    ],
}
```

### Test Discovery and Execution

The `pattern_tests/` directory contains:

- **`discovery.py`**: Scans pattern files and loads test metadata
- **`test_e2e_generated.py`**: Pytest tests for on-device E2E validation
- **`test_lit_generated.py`**: In-process LIT-style tests (no FileCheck needed)
- **`lit_generator.py`**: Generates standalone LIT test files

## Test Metadata Format

### `PATTERN_TEST_METADATA` Structure

```python
{
    "pattern_name": str,        # Unique identifier for the pattern
    "description": str,          # Human-readable description

    "lit_tests": [              # List of LIT-style rewrite tests
        {
            "name": str,        # Test function/case name
            "module_text": str, # MLIR module to parse and transform
            "file_checks": [str], # FileCheck directive strings
            "description": str, # Optional test description
        }
    ],

    "e2e_tests": [              # List of on-device E2E tests
        {
            "name": str,        # Test function name
            "description": str, # Test description
            "kernel_fn": callable, # The @d2m.kernel function to test
            "input_generator": callable, # () -> dict[str, Tensor]
            "reference_fn": callable, # (**inputs) -> Tensor
            "layout_config": dict, # Layout constructor kwargs
            "kernel_args": dict,   # Additional kernel call kwargs
        }
    ],
}
```

### Field Descriptions

#### `lit_tests`

- **`name`**: Identifier for this test case
- **`module_text`**: MLIR module IR as a string
- **`file_checks`**: List of FileCheck directives (e.g., `"CHECK: d2m.generic"`)
- **`description`**: Optional explanation of what this test validates

#### `e2e_tests`

- **`name`**: Test function name (should start with `test_`)
- **`description`**: Human-readable description shown in test output
- **`kernel_fn`**: Reference to the `@d2m.kernel` decorated function
- **`input_generator`**: Lambda returning dict of input tensors
- **`reference_fn`**: Lambda computing expected output from inputs
- **`layout_config`**: Dict of kwargs for `d2m.Layout()` constructor
- **`kernel_args`**: Dict of additional kwargs passed to kernel (e.g., `m_blocks`, `grid`)

## Running Tests

### Run all pattern E2E tests (pytest)
```bash
pytest test/d2m-jit/pattern_tests/test_e2e_generated.py
```

### Run all pattern LIT-style tests (pytest, in-process)
```bash
pytest test/d2m-jit/pattern_tests/test_lit_generated.py
```

### Generate standalone LIT test files
```bash
python -m test.d2m_jit.pattern_tests.lit_generator
```

This generates files in `test/d2m-jit/lit_generated/` that can be run with the standard LIT test runner.

### Run a specific pattern's tests
```bash
# E2E tests for eltwise_exp pattern
pytest test/d2m-jit/pattern_tests/test_e2e_generated.py -k "eltwise_exp"

# LIT tests for eltwise_add_exp pattern
pytest test/d2m-jit/pattern_tests/test_lit_generated.py -k "eltwise_add_exp"
```

## Benefits of This Approach

1. **Single Source of Truth**: Pattern implementation and tests live in the same file
2. **DRY**: No duplication between LIT test modules and E2E test modules
3. **Automatic Discovery**: New patterns are automatically tested when metadata is added
4. **Flexible Execution**: Run as pytest (fast iteration) or LIT (CI integration)
5. **Easy to Extend**: Adding a new pattern test is just adding metadata to the pattern file

## Migration Guide

To migrate an existing pattern to the new testing framework:

1. **Add metadata to the pattern file** (`tools/d2m-jit/patterns/my_pattern.py`):
   ```python
   PATTERN_TEST_METADATA = {
       "pattern_name": "my_pattern",
       "description": "...",
       "lit_tests": [...],
       "e2e_tests": [...],
   }
   ```

2. **Copy test data from old files**:
   - LIT test module text from `test/d2m-jit/lit/my_pattern.py`
   - E2E test logic from `test/d2m-jit/test_pattern_*.py`

3. **Verify tests run**:
   ```bash
   pytest test/d2m-jit/pattern_tests/ -k "my_pattern"
   ```

4. **Optional: Delete old test files** once migration is complete

## Examples

See the updated pattern files for reference:
- `tools/d2m-jit/patterns/eltwise_exp_to_kernel.py`
- `tools/d2m-jit/patterns/eltwise_add_exp_to_kernel.py`
