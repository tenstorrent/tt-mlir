# Pattern Test YAML Configuration Guide

## Overview

Pattern tests can now be defined in external YAML files instead of embedding them in pattern `.py` files. This provides:

- **Clean separation**: Pattern code stays pure, tests live in `.test.yaml` files
- **Easy editing**: Modify test configs without touching Python code
- **Version control friendly**: Better diffs for test config changes
- **Tooling support**: CLI tools for generating, validating, and managing configs

## Quick Start

### 1. Generate a YAML Template

```bash
cd test/d2m-jit/pattern_tests
python yaml_cli.py generate my_pattern my_pattern_to_kernel
```

This creates `tools/d2m-jit/patterns/my_pattern_to_kernel.test.yaml` with a template.

### 2. Edit the YAML File

```yaml
pattern_name: my_pattern
pattern_module: my_pattern_to_kernel
description: What this pattern does

lit_tests:
  - name: my_pattern_positive
    module_text: |
      module {
        # Your MLIR module here
      }
    file_checks:
      - "CHECK: d2m.generic"

e2e_tests:
  - name: test_my_pattern_on_device
    kernel: my_kernel_function
    inputs:
      - name: x
        shape: [32, 32]
        generator: uniform
    reference: "torch.exp(x)"
    layout:
      shape: [32, 32]
    kernel_args:
      m_blocks: 1
      n_blocks: 1
      grid: [1, 1]
```

### 3. Validate the Config

```bash
python yaml_cli.py validate my_pattern_to_kernel.test.yaml
```

### 4. Run the Tests

```bash
pytest test_e2e_generated.py -k "my_pattern"
pytest test_lit_generated.py -k "my_pattern"
```

## YAML Schema Reference

### Top Level

```yaml
pattern_name: string          # Unique identifier
pattern_module: string         # Python module name (without .py)
description: string            # Multi-line description
tags: [string]                 # Optional tags for filtering
lit_tests: [LitTest]          # List of LIT tests
e2e_tests: [E2ETest]          # List of E2E tests
```

### LIT Test

```yaml
- name: string                 # Test identifier
  description: string          # What this validates (optional)
  module_text: |               # MLIR module IR
    module {
      # Your IR here
    }
  file_checks:                 # FileCheck patterns
    - "CHECK-LABEL: ..."
    - "CHECK: ..."
    - "CHECK-NOT: ..."
```

### E2E Test

```yaml
- name: string                 # Test function name
  description: string          # What this validates (optional)
  kernel: string               # Kernel function name from pattern module

  inputs:                      # List of input specs
    - name: string             # Parameter name
      shape: [int]             # Tensor shape
      generator: string        # "uniform", "normal", "randn", "ones", "zeros"
      dtype: string            # "float32", "float16", etc.
      range_min: float         # For uniform generator
      range_max: float         # For uniform generator

  reference: string            # Python expression (e.g., "torch.exp(x)")

  layout:                      # d2m.Layout config
    shape: [int]
    dtype: string
    block_shape: [int]
    grid_shape: [int]
    tiled: bool
    memory_space: string       # "L1" or "DRAM"

  kernel_args:                 # Additional kwargs for kernel
    m_blocks: int
    n_blocks: int
    grid: [int]

  pcc_threshold: float         # Default: 0.99
  seed: int                    # Random seed, default: 0
```

### Input Generators

**Uniform distribution:**
```yaml
generator: uniform
range_min: -1.0
range_max: 1.0
```

**Normal distribution:**
```yaml
generator: normal
mean: 0.0
std: 1.0
```

**Standard normal:**
```yaml
generator: randn
```

**Constants:**
```yaml
generator: ones   # or zeros, arange
```

### Reference Expressions

Python expressions that compute expected output from inputs:

```yaml
# Single input
reference: "torch.exp(x)"

# Multiple inputs
reference: "torch.exp(a + b)"

# Complex expressions
reference: "torch.matmul(a, b.transpose(-2, -1))"
```

Input names must match the `inputs[].name` fields.

## CLI Commands

### Generate Template

```bash
python yaml_cli.py generate <pattern_name> <pattern_module> [-o OUTPUT] [-f]
```

**Options:**
- `-o, --output`: Specify output file path
- `-f, --force`: Overwrite existing file

### Validate Configs

```bash
# Validate specific file
python yaml_cli.py validate my_pattern.test.yaml

# Validate all configs
python yaml_cli.py validate
```

### List Configs

```bash
python yaml_cli.py list
```

Shows all discovered YAML configs with summary info.

## Examples

See working examples:
- `tools/d2m-jit/patterns/eltwise_exp_to_kernel.test.yaml`
- `tools/d2m-jit/patterns/eltwise_add_exp_to_kernel.test.yaml`

## Migration from Dict-Based Configs

If you have existing `PATTERN_TEST_METADATA` dicts in pattern files:

1. Generate YAML template:
   ```bash
   python yaml_cli.py generate pattern_name pattern_module
   ```

2. Copy test data from dict to YAML file

3. Validate:
   ```bash
   python yaml_cli.py validate pattern_module.test.yaml
   ```

4. Run tests to verify:
   ```bash
   pytest test_e2e_generated.py -k "pattern_name"
   ```

5. **Optional**: Remove `PATTERN_TEST_METADATA` from pattern file

Both formats work simultaneously. YAML configs take precedence if both exist for the same pattern.

## Benefits of YAML Configs

1. **Cleaner pattern files**: Focus on pattern logic only
2. **Non-Python users**: Anyone can edit test configs
3. **Better diffs**: Config changes don't mix with code changes
4. **Tooling**: Easy to generate, validate, and transform configs
5. **Scalability**: Can generate configs from other sources (traces, etc.)
6. **Documentation**: YAML is self-documenting with comments

## Troubleshooting

**YAML syntax error:**
```
python yaml_cli.py validate my_pattern.test.yaml
```

**Kernel not found:**
- Ensure `kernel:` value matches function name in pattern module
- Check `pattern_module:` is correct

**Input name mismatch:**
- Input names in `inputs:` must match variables in `reference:` expression

**Import errors:**
- Ensure pattern module is valid Python
- Check for syntax errors in pattern file
