# D2M-JIT Pattern Testing Refactoring

## Summary

This refactoring consolidates d2m-jit pattern testing by moving test metadata from separate test files into the pattern definition files themselves. This creates a single source of truth and enables automatic test generation.

## Before: Distributed Test Files

### Old Structure
```
tools/d2m-jit/patterns/
  eltwise_exp_to_kernel.py           # Pattern implementation only
  eltwise_add_exp_to_kernel.py       # Pattern implementation only

test/d2m-jit/
  test_pattern_eltwise.py            # E2E tests (hardcoded)
  lit/
    eltwise_exp_pattern.py           # LIT test (hardcoded module)
    eltwise_add_exp_pattern.py       # LIT test (hardcoded module)
```

### Problems with Old Approach

1. **Duplication**: Test modules and golden data scattered across multiple files
2. **Manual Sync**: Changes to patterns require updating 3+ separate files
3. **No Discovery**: Tests must be manually written for each pattern
4. **Maintenance Burden**: Adding a new pattern means creating multiple test files

## After: Consolidated Pattern Files

### New Structure
```
tools/d2m-jit/patterns/
  eltwise_exp_to_kernel.py           # Pattern + test metadata (all in one)
  eltwise_add_exp_to_kernel.py       # Pattern + test metadata (all in one)

test/d2m-jit/pattern_tests/
  discovery.py                       # Discovers pattern test metadata
  test_e2e_generated.py              # Auto-generated E2E tests
  test_lit_generated.py              # Auto-generated LIT tests (in-process)
  lit_generator.py                   # Generates standalone LIT files
  conftest.py                        # Pytest configuration
  README.md                          # Documentation
```

### Benefits of New Approach

1. **Single Source of Truth**: Pattern code and tests in one file
2. **Automatic Discovery**: Framework finds and runs all tests automatically
3. **DRY Principle**: No duplication between test files
4. **Easy to Add Patterns**: Just add `PATTERN_TEST_METADATA` to pattern file
5. **Flexible Testing**: Run as pytest (fast) or generate LIT files (CI)

## Pattern File Changes

### Example: `eltwise_exp_to_kernel.py`

```python
# Pattern implementation (unchanged)
@d2m.kernel
def exp_fused(in_t, out_t, m_blocks, n_blocks):
    # ... kernel code ...

@d2m.pattern(root=ttir.ExpOp, benefit=10)
def lower_exp(op, rewriter):
    # ... pattern code ...

# NEW: Test metadata added to pattern file
PATTERN_TEST_METADATA = {
    "pattern_name": "eltwise_exp",
    "description": "Single-eltwise pattern: ttir.exp -> d2m subgraph",

    "lit_tests": [
        {
            "name": "exp_pattern_positive",
            "module_text": """...""",  # From old lit/eltwise_exp_pattern.py
            "file_checks": [...],       # From old lit/eltwise_exp_pattern.py
        }
    ],

    "e2e_tests": [
        {
            "name": "test_pattern_exp_kernel_on_device",
            "kernel_fn": exp_fused,
            "input_generator": lambda: {...},    # From old test_pattern_eltwise.py
            "reference_fn": lambda x: torch.exp(x),
            "layout_config": {...},
            "kernel_args": {...},
        }
    ],
}
```

## Test Framework Components

### 1. Discovery (`discovery.py`)

Automatically finds and loads `PATTERN_TEST_METADATA` from pattern files:

```python
from test.d2m_jit.pattern_tests.discovery import discover_all_pattern_tests

all_patterns = discover_all_pattern_tests()
# Returns list of metadata dicts for all patterns
```

### 2. E2E Tests (`test_e2e_generated.py`)

Parametrized pytest tests that run on device:

```python
@pytest.mark.parametrize("pattern_metadata,e2e_test_config", ...)
def test_pattern_kernel_e2e(pattern_metadata, e2e_test_config):
    # Generates inputs, runs kernel on device, asserts PCC
```

Run with: `pytest test/d2m-jit/pattern_tests/test_e2e_generated.py`

### 3. LIT Tests (`test_lit_generated.py`)

In-process LIT-style tests (no FileCheck binary needed):

```python
@pytest.mark.parametrize("pattern_metadata,lit_test_config", ...)
def test_pattern_lit_style(pattern_metadata, lit_test_config):
    # Parses module, applies patterns, checks output
```

Run with: `pytest test/d2m-jit/pattern_tests/test_lit_generated.py`

### 4. LIT File Generator (`lit_generator.py`)

Generates standalone LIT test files for CI:

```bash
python -m test.d2m_jit.pattern_tests.lit_generator
# Generates test/d2m-jit/lit_generated/eltwise_exp_pattern_generated.py
```

## Migration Path

### For Each Pattern:

1. **Add metadata to pattern file**:
   - Copy module text from `test/d2m-jit/lit/*.py`
   - Copy test logic from `test/d2m-jit/test_pattern_*.py`
   - Structure as `PATTERN_TEST_METADATA` dict

2. **Verify tests run**:
   ```bash
   pytest test/d2m-jit/pattern_tests/ -k "pattern_name"
   ```

3. **Optional: Remove old test files** after validation

### Example Migration Checklist:

- [x] `eltwise_exp_to_kernel.py` - Added metadata
- [x] `eltwise_add_exp_to_kernel.py` - Added metadata
- [ ] Old test files can be kept for comparison or removed

## Running Tests

### All pattern tests (pytest)
```bash
# E2E tests (on-device)
pytest test/d2m-jit/pattern_tests/test_e2e_generated.py

# LIT tests (in-process)
pytest test/d2m-jit/pattern_tests/test_lit_generated.py

# All pattern tests
pytest test/d2m-jit/pattern_tests/
```

### Specific pattern tests
```bash
# Just eltwise_exp
pytest test/d2m-jit/pattern_tests/ -k "eltwise_exp"

# Just E2E tests for add_exp
pytest test/d2m-jit/pattern_tests/test_e2e_generated.py -k "add_exp"
```

### Generate LIT files for CI
```bash
python -m test.d2m_jit.pattern_tests.lit_generator
# Outputs to test/d2m-jit/lit_generated/
```

## Backwards Compatibility

The old test files (`test_pattern_eltwise.py`, `lit/eltwise_exp_pattern.py`, etc.) continue to work unchanged. The new framework is additive and can run alongside existing tests during migration.

## Future Enhancements

Possible extensions to the framework:

1. **Auto-generate golden data**: Save/load inputs and outputs automatically
2. **Pattern composition tests**: Test combinations of patterns
3. **Performance benchmarks**: Add perf test metadata
4. **Negative tests**: Add metadata for expected failures
5. **Cross-backend tests**: Test patterns on multiple devices/backends
