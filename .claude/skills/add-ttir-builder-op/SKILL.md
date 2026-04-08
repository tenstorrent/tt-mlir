---
name: add-ttir-builder-op
description: Add full builder API support (@tag, @parse, @split) for a TTIR op. Use this skill whenever the user wants to add builder support for a new TTIR op, upgrade an existing _op_proxy-based op to use @tag/@parse/@split decorators, or asks about how to add builder API for an op in ttir_builder.py. Also trigger when the user mentions adding tag/parse/split for an op, or wants to make an op work with the parse/split test infrastructure.
---

# Adding TTIR Builder Op Support

This skill walks through adding full `@tag`, `@parse`, and `@split` builder support for a TTIR op. The process touches 3 files and creates 1 test file.

## Overview of Files

| File | Purpose |
|------|---------|
| `tools/builder/ttir/ttir_builder.py` | Builder methods (`@tag`, `@parse`, `@split`) |
| `tools/golden/mapping.py` | Golden function mappings (torch reference implementations) |
| `test/python/golden/mlir_snippets/ttir/ttir_<op_name>.mlir` | MLIR snippet for parse/split tests |
| `include/ttmlir/Dialect/TTIR/IR/TTIROps.td` | Op definitions (read-only reference) |

## Step 1: Verify the Op and Find a Reference

Check `include/ttmlir/Dialect/TTIR/IR/TTIROps.td` for the op definition to understand its type (unary, binary, reduction, convolution, etc.) and any special attributes.

Then find an existing op of the **same type** that already has full `@tag`/`@parse`/`@split` support in `tools/builder/ttir/ttir_builder.py`. This is your reference implementation. Some examples:

- **Unary elementwise** (1 input): `sigmoid` (~line 7591), `cos` (~line 2459)
- **Binary elementwise** (2 inputs): `add` (~line 7467), `multiply` (~line 7077)
- **Ops with extra attributes**: `sort` (~line 3689), `rearrange` (~line 1083)
- **Reduction ops**: `sum` (~line 7325), `reduce_and` (~line 1208)
- **Multi-output ops**: `max_pool2d_with_indices` (~line 4224)
- **Conv/matmul ops**: `conv2d` (~line 11102), `matmul` (~line 12349)

Search for `@tag(ttir.` in `ttir_builder.py` to find all ops with full support. Pick the closest match to your new op and use it as a template throughout the remaining steps.

## Step 2: Ensure Golden Function Has Correct Signature

Check `tools/golden/mapping.py` for the op's entry in the golden mapping dict.

**The golden function MUST match how the `@tag` method calls it.** The `@tag` method passes all input golden tensors followed by `mlir_output_type` as the last argument. If the mapping points directly to a bare torch function (e.g., `torch.nn.functional.gelu`), it will fail at runtime because those don't accept an output type parameter.

### Fix: Create a wrapper golden function

Look at how the reference op's golden function is defined in `tools/golden/mapping.py` and follow the same pattern. The wrapper calls the underlying torch function and converts the output dtype:

```python
def ttir_<op_name>_golden(
    input_tensor: GoldenMapTensor, ..., output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.<op_function>(input_tensor, ...).to(output_dtype)
```

The number and type of parameters before `output_type_mlir` depends on the op — match your reference op's golden function signature.

Then update the mapping dict to point to the new wrapper.

If the golden function already has the correct signature, no changes are needed.

## Step 3: Add @tag, @parse, @split Methods

In `tools/builder/ttir/ttir_builder.py`, replace the existing `_op_proxy` method (if any) or add new methods. Follow your reference op's implementation closely, substituting:
- The op class name (e.g., `SigmoidOp` -> `<OpName>Op`)
- Method names (e.g., `sigmoid` -> `<op_name>`)
- Module/builder variable names in split (e.g., `sigmoid_module` -> `<op_name>_module`)
- Input accessors in parse/split (unary uses `old_op.input`, binary uses `old_op.lhs`/`old_op.rhs` — check your reference)

Key points that apply to all op types:
- The `@tag` method must return `OpResult` (not `OpView`)
- `get_opview_from_method`, `get_opview_from_parser`, `get_opview_from_split` retrieve the MLIR op class from the decorator metadata
- Golden tensors must be computed and set via `_set_golden_tensor` in all three methods
- Input operands must be presharded if necessary via `_annotate_presharded_arg` in all `@split` methods
- The `@split` method creates an isolated `Module` with its own `TTIRBuilder` instance

## Step 4: Add MLIR Snippet for Parse/Split Tests

Create `test/python/golden/mlir_snippets/ttir/ttir_<op_name>.mlir`.

The `test_parse_split_ops.py` test auto-discovers all `.mlir` files in this directory — no test code changes needed.

Look at an existing snippet for a similar op to get the right format. The snippet should define a minimal `module` with a `func.func` that exercises the op. For example, find existing snippets with:
```bash
ls test/python/golden/mlir_snippets/ttir/
```

## Step 5: Verify Test Wrapper Compatibility

Check the relevant test file (e.g., `test/python/golden/ttir_ops/eltwise/test_ttir_unary.py` for elementwise ops) for the existing test wrapper. Since `output_type` and `loc` are added as optional parameters, existing wrappers that call `builder.<op_name>(in0, unit_attrs=unit_attrs)` remain backward-compatible.

If no test wrapper exists yet, add one following the pattern of nearby ops in the same test file.

## Step 6: Build and Test

```bash
# Rebuild (copies source to build/python_packages/)
cmake --build build

# Test parse/split (auto-discovered from MLIR snippet)
pytest test/python/golden/test_parse_split_ops.py -k "ttir_<op_name>"

# Test builder (adjust test file path based on op type)
export SYSTEM_DESC_PATH=$(pwd)/ttrt-artifacts/system_desc.ttsys
pytest test/python/golden/ttir_ops/eltwise/test_ttir_unary.py -k "<op_name>"
```

The rebuild step is essential because tests run against `build/python_packages/`, not the source files.

## Checklist

- [ ] Op exists in `TTIROps.td` — identified op type and found reference implementation
- [ ] Golden function in `tools/golden/mapping.py` accepts `output_type_mlir` as last arg
- [ ] `@tag` method added in `ttir_builder.py` (returns `OpResult`, not `OpView`)
- [ ] `@parse` method added in `ttir_builder.py`
- [ ] `@split` method added in `ttir_builder.py`
- [ ] MLIR snippet created in `test/python/golden/mlir_snippets/ttir/`
- [ ] `cmake --build build` run to install changes
- [ ] Parse/split tests pass
- [ ] Builder tests pass
