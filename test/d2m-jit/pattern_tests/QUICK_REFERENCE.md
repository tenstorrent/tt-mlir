# Pattern Testing Quick Reference

## File Structure

```
tools/d2m-jit/patterns/
  my_pattern.py              ← Pattern implementation + PATTERN_TEST_METADATA

test/d2m-jit/pattern_tests/
  discovery.py               ← Discovers patterns with test metadata
  test_e2e_generated.py      ← Auto-generated E2E tests (pytest)
  test_lit_generated.py      ← Auto-generated LIT tests (pytest)
  lit_generator.py           ← Generates standalone LIT files
  conftest.py                ← Pytest configuration
  README.md                  ← Full documentation
  PATTERN_TEMPLATE.py        ← Template for new patterns
```

## Adding Tests to a Pattern

In your pattern file (`tools/d2m-jit/patterns/my_pattern.py`):

```python
import torch
import d2m_jit as d2m
from ttmlir.dialects import ttir

# 1. Define kernel
@d2m.kernel
def my_kernel(in_t, out_t, m_blocks, n_blocks):
    # ... kernel implementation ...
    pass

# 2. Define pattern
@d2m.pattern(root=ttir.SomeOp, benefit=10)
def my_pattern(op, rewriter):
    # ... pattern implementation ...
    pass

# 3. Add test metadata
PATTERN_TEST_METADATA = {
    "pattern_name": "my_pattern",
    "description": "What this pattern does",

    "lit_tests": [
        {
            "name": "test_case_name",
            "module_text": """module { ... }""",
            "file_checks": ["CHECK: ...", "CHECK-NOT: ..."],
        }
    ],

    "e2e_tests": [
        {
            "name": "test_my_kernel_on_device",
            "kernel_fn": my_kernel,
            "input_generator": lambda: {"x": torch.rand(32, 32)},
            "reference_fn": lambda x: torch.exp(x),
            "layout_config": {
                "shape": (32, 32),
                "dtype": d2m.float32,
                "block_shape": [1, 1],
                "grid_shape": [1, 1],
                "tiled": True,
            },
            "kernel_args": {"m_blocks": 1, "n_blocks": 1, "grid": (1, 1)},
        }
    ],
}
```

## Running Tests

```bash
# Validate pattern discovery
cd test/d2m-jit/pattern_tests
python3 validate_refactoring.py

# Run all pattern tests
pytest test/d2m-jit/pattern_tests/

# Run specific pattern tests
pytest test/d2m-jit/pattern_tests/ -k "my_pattern"

# Run only E2E tests
pytest test/d2m-jit/pattern_tests/test_e2e_generated.py

# Run only LIT tests
pytest test/d2m-jit/pattern_tests/test_lit_generated.py

# Generate standalone LIT files
python -m test.d2m_jit.pattern_tests.lit_generator
```

## Metadata Reference

### `lit_tests` Entry

```python
{
    "name": str,              # Test name/identifier
    "module_text": str,       # MLIR module (before rewrite)
    "file_checks": [str],     # FileCheck patterns
    "description": str,       # Optional: what this tests
}
```

**FileCheck Patterns:**
- `"CHECK: pattern"` - Must appear in output
- `"CHECK-NOT: pattern"` - Must NOT appear
- `"CHECK-LABEL: pattern"` - Section marker (resets position)
- Use `{{.*}}` for wildcards (converted to regex)

### `e2e_tests` Entry

```python
{
    "name": str,              # Test function name
    "description": str,       # Test description
    "kernel_fn": callable,    # The @d2m.kernel function
    "input_generator": callable,  # () -> dict[str, Tensor]
    "reference_fn": callable, # (**inputs) -> Tensor
    "layout_config": dict,    # kwargs for d2m.Layout()
    "kernel_args": dict,      # kwargs for kernel call
}
```

**Kernel Signature:**
```python
kernel_fn(*inputs, output, **kernel_args)
```

**Input Generator:**
```python
lambda: {
    "param1": torch.rand(32, 32),
    "param2": torch.ones(32, 32),
}
```

**Reference Function:**
```python
lambda param1, param2: torch.exp(param1 + param2)
```

## Common Patterns

### Single Input, Single Output

```python
"e2e_tests": [{
    "kernel_fn": exp_fused,
    "input_generator": lambda: {"x": torch.rand(32, 32)},
    "reference_fn": lambda x: torch.exp(x),
    "layout_config": {...},
    "kernel_args": {"m_blocks": 1, "n_blocks": 1, "grid": (1, 1)},
}]
```

### Multiple Inputs

```python
"e2e_tests": [{
    "kernel_fn": add_exp_fused,
    "input_generator": lambda: {
        "a": torch.rand(32, 32),
        "b": torch.rand(32, 32),
    },
    "reference_fn": lambda a, b: torch.exp(a + b),
    # ...
}]
```

### Numerically Stable Inputs

```python
"input_generator": lambda: {
    "x": (torch.rand(32, 32) - 0.5) * 2.0,  # range: [-1, 1]
}
```

### Multiple Test Cases

```python
"e2e_tests": [
    {
        "name": "test_small_grid",
        # ... config for 1x1 grid ...
    },
    {
        "name": "test_large_grid",
        # ... config for 4x4 grid ...
    },
]
```

## Examples

See working examples in:
- `tools/d2m-jit/patterns/eltwise_exp_to_kernel.py`
- `tools/d2m-jit/patterns/eltwise_add_exp_to_kernel.py`
- `test/d2m-jit/pattern_tests/PATTERN_TEMPLATE.py` (annotated template)

## Troubleshooting

**Pattern not discovered:**
- Check `PATTERN_TEST_METADATA` is at module level (not indented)
- Run validation: `python3 validate_refactoring.py`

**LIT test fails:**
- Print module: add `print(mod)` in your test
- Check FileCheck patterns match exactly (case-sensitive)
- Use `CHECK-NOT` to verify old ops are removed

**E2E test fails:**
- Check input shapes match layout config
- Verify reference function uses same inputs as generator
- Check kernel_args match kernel signature

**Import errors:**
- Ensure d2m_jit environment is set up
- Check Python path includes test directory
