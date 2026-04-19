# Task

Create a pytest test in `tools/chisel/tests/test_golden_execution.py` that loads a flatbuffer binary, iterates over every TTNN op in every program, generates random input tensors matching each op's input shapes and element types, and calls `execute_golden()` on each one. Ops with no golden implementation should be skipped (not failed). The test should report how many ops were executed vs skipped.

This test is analogous to `tools/chisel/tests/test_ir_module.py` (which validates op walk order and debug info against the flatbuffer), but instead of structural validation it actually executes each op through the golden system.

## Context from current session

**Branch**: `ndrakulic/chisel-executor-golden`
**Base repo**: `/localdev/ndrakulic/tt-mlir`

---

## Reference: test_ir_module.py (from ndrakulic/chisel-ir-module branch)

This is the pattern to follow for loading the binary and iterating programs/ops:

```python
# tools/chisel/tests/test_ir_module.py
import pytest
import ttrt.binary
from chisel.ops import IRModule

@pytest.fixture
def binary(binary_path):
    return ttrt.binary.load_binary_from_path(binary_path)

@pytest.fixture
def ir_module(binary):
    mlir_json = ttrt.binary.mlir_as_dict(binary)
    functions = [binary.get_program_name(i) for i in range(binary.get_num_programs())]
    return IRModule(mlir_source=mlir_json["source"], functions=functions)

def _iterate_programs(binary):
    for i in range(binary.get_num_programs()):
        yield i, binary.get_program_name(i)

def test_ops(ir_module, binary):
    for prog_idx, prog_name in _iterate_programs(binary):
        mlir_ops = ir_module.get_function_ops(prog_name)
        fb_ops = ttrt.binary.program_ops_as_dict(binary, prog_idx)
        for i, (mlir_op, fb_op) in enumerate(zip(mlir_ops, fb_ops, strict=True)):
            ...
```

## Reference: conftest.py (shared fixture for --binary CLI option)

```python
# tools/chisel/tests/conftest.py
def pytest_addoption(parser):
    parser.addoption("--binary", help="Path to a .ttnn file or directory.")

def pytest_generate_tests(metafunc):
    if "binary_path" in metafunc.fixturenames:
        paths = _collect_binary_paths(metafunc.config)
        metafunc.parametrize("binary_path", paths)
```

Run with: `pytest tools/chisel/tests/test_golden_execution.py --binary path/to/file.ttnn`

---

## Reference: executor.py (what execute_golden does)

```python
# tools/chisel/chisel/executor.py
from chisel.ops import IRModule, get_op_inputs
from golden import get_chisel_golden_function, GoldenMapTensor

def execute_golden(op, ir_module, function_name, inputs: dict) -> torch.Tensor:
    """
    inputs: Dict[str, torch.Tensor] — keyed by SSA name (e.g. '%arg0')
    Returns: torch.Tensor — the golden output
    Raises: RuntimeError if no golden for the op type
    """
    golden_fn = get_chisel_golden_function(type(op))
    if golden_fn is None:
        raise RuntimeError(f"No golden implementation for {type(op).__name__}")

    op_inputs = get_op_inputs(op)
    asm_state = ir_module.get_asm_state(function_name)
    golden_inputs = {
        inp.get_name(asm_state): GoldenMapTensor({0: inputs[inp.get_name(asm_state)]}, (1, 1))
        for inp in op_inputs
    }
    result = golden_fn(op, golden_inputs, asm_state)
    ...
```

The `inputs` dict is keyed by SSA name. You need to:
1. Get `op_inputs = get_op_inputs(op)` — these are the operands
2. For each operand, get its SSA name via `inp.get_name(asm_state)`
3. Generate a random `torch.Tensor` matching the operand's shape and dtype
4. Pass the dict to `execute_golden()`

---

## How to get tensor shape/dtype from an operand

Each operand is an MLIR `Value`. Its type is a `RankedTensorType`. Example:

```python
from ttmlir.ir import RankedTensorType
from chisel.ops import get_op_inputs

op_inputs = get_op_inputs(op)
for operand in op_inputs:
    ranked_type = RankedTensorType(operand.type)
    shape = list(ranked_type.shape)
    # element_type is an mlir type — map it to torch dtype
    element_type = ranked_type.element_type
```

Map element types to torch dtypes (check `ttmlir.ir` for exact type classes):
- `F32Type` / `f32` → `torch.float32`
- `BF16Type` / `bf16` → `torch.bfloat16`
- `F16Type` / `f16` → `torch.float16`
- `IntegerType` width 32 → `torch.int32`
- `IntegerType` width 16 → `torch.int16`
- etc.

Generate random tensor: `torch.randn(shape, dtype=dtype)` for float types, `torch.randint(0, 10, shape, dtype=dtype)` for int types.

---

## What the test should do

```
for each program in binary:
    ir_module ops = ir_module.get_function_ops(prog_name)
    for each op:
        try:
            build random inputs dict keyed by SSA name
            call execute_golden(op, ir_module, prog_name, inputs)
            record: executed
        except RuntimeError (no golden):
            record: skipped
        except Exception:
            record: failed (with error)

report: X executed, Y skipped, Z failed
assert Z == 0  (or parametrize to be lenient)
```

---

## Files to look at in this worktree

- `tools/chisel/chisel/executor.py` — `execute_golden()` implementation
- `tools/chisel/chisel/ops.py` — `IRModule`, `get_op_inputs()`
- `tools/chisel/tests/conftest.py` — existing fixtures (already has `--binary` support)
- `tools/chisel/tests/test_ir_module.py` — structural test to follow as pattern
- `tools/chisel/tests/test_executor.py` — unit tests for executor (for reference)
- `lib/Dialect/TTNN/IR/TTNNOps.cpp` or similar — if you need to understand op types

## Notes

- `get_chisel_golden_function(type(op))` returns `None` if no golden exists — catch `RuntimeError` or check before calling
- The `conftest.py` already handles `--binary` and parametrizes `binary_path` — reuse it
- Keep the test file in `tools/chisel/tests/test_golden_execution.py`
- Add SPDX header: `# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC` / `# SPDX-License-Identifier: Apache-2.0`
