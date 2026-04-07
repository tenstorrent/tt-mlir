# PR 1: Single Op Isolation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver op-level isolation testing for Chisel — each TTNN op is tested independently against its golden (CPU) reference, proving the core comparison loop works without cross-op chaining.

**Architecture:** Chisel hooks into TTRT execution via `DebugHooks` callbacks. `preOp` copies device input tensors to host, `postOp` runs the golden function via `GOLDEN_MAPPINGS` and compares. A slim `ChiselContext` singleton holds the MLIR module, op iterator, and stashed inputs. Golden functions are called generically using signature introspection to match MLIR attributes to function parameters.

**Tech Stack:** Python, MLIR Python bindings (`ttmlir.ir`, `ttmlir.dialects.ttnn`), `tools/golden` (GOLDEN_MAPPINGS, GoldenMapTensor), `ttrt.runtime` (DebugHooks, tensor access), PyTorch, pytest.

---

## File Structure

### New Files

| File | Responsibility |
|------|----------------|
| `tools/golden/metrics.py` | Pure-torch PCC/atol/rtol comparison functions (resolves TODO, removes PR 0c dependency) |
| `tools/chisel/CMakeLists.txt` | CMake packaging via `declare_mlir_python_sources` |
| `tools/chisel/chisel/__init__.py` | Package init + public exports |
| `tools/chisel/chisel/utils.py` | Dtype maps, runtime tensor → torch conversion, debug_wrap decorator |
| `tools/chisel/chisel/ops.py` | `IRModule` wrapper, `get_op_inputs()`, `get_op_outputs()` |
| `tools/chisel/chisel/executor.py` | `execute_golden()` — CPU replay of a single TTNN op via GOLDEN_MAPPINGS |
| `tools/chisel/chisel/context.py` | Slim `ChiselContext` singleton (ir_module, op_iter, stashed inputs) |
| `tools/chisel/chisel/callbacks.py` | `chisel_pre_op_callback` / `chisel_post_op_callback` for DebugHooks |
| `tools/chisel/tests/test_utils.py` | Tests for dtype maps and tensor conversion |
| `tools/chisel/tests/test_ops.py` | Tests for IRModule and op extraction |
| `tools/chisel/tests/test_executor.py` | Tests for execute_golden |
| `tools/chisel/tests/test_context.py` | Tests for ChiselContext singleton lifecycle |
| `tools/chisel/tests/test_callbacks.py` | Tests for preOp/postOp callbacks |

### Modified Files

| File | Change |
|------|--------|
| `tools/golden/CMakeLists.txt` | Add `metrics.py` to `GoldenSources` |
| `tools/golden/__init__.py` | Export metrics functions |
| `tools/CMakeLists.txt` | Add `add_subdirectory(chisel)` |

---

## Task 1: Golden Metrics Module

**Why:** The PR 1 doc has a TODO: _"Lets just copy metrics from builder into tools/golden/metrics rewriten in torch and not numpy."_ This removes the PR 0c dependency and unblocks everything else. The old chisel's `runtime/tools/chisel/chisel/utils/metrics.py` is already pure-torch — port it directly.

**Files:**
- Create: `tools/golden/metrics.py`
- Create: `tools/chisel/tests/test_metrics.py`
- Modify: `tools/golden/CMakeLists.txt`
- Modify: `tools/golden/__init__.py`

- [ ] **Step 1: Create `tools/golden/metrics.py`**

Port from `runtime/tools/chisel/chisel/utils/metrics.py` (already pure-torch):

```python
# tools/golden/metrics.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unified tensor comparison metrics — PCC, absolute error, relative error.

Pure torch, no numpy dependency. Shapes are expected to match (single-module
TTNN means golden and device tensors have the same shape by construction).
"""
import torch


def compute_atol(golden: torch.Tensor, calculated: torch.Tensor) -> float:
    """Max absolute difference: max(|golden - calculated|)."""
    golden = golden.to(torch.float32)
    calculated = calculated.to(torch.float32)

    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
        return 0.0

    if torch.all(torch.isinf(golden)) and torch.all(torch.isinf(calculated)):
        if torch.all(golden == calculated):
            return 0.0
        return torch.inf

    return torch.max(torch.abs(golden - calculated)).item()


def compute_rtol(golden: torch.Tensor, calculated: torch.Tensor) -> float:
    """Max relative difference: max(|(golden - calculated) / (golden + eps)|)."""
    golden = golden.to(torch.float32)
    calculated = calculated.to(torch.float32)

    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
        return 0.0

    if torch.all(torch.isinf(golden)) and torch.all(torch.isinf(calculated)):
        if torch.all(golden == calculated):
            return 0.0
        return torch.inf

    return torch.max(torch.abs((golden - calculated) / (golden + 1e-8))).item()


def compute_pcc(golden: torch.Tensor, calculated: torch.Tensor) -> float:
    """Pearson correlation coefficient between two tensors.

    Returns 1.0 for identical tensors, 0.0 for completely different.
    """
    x = golden.to(torch.float32).flatten()
    y = calculated.to(torch.float32).flatten()

    if torch.all(torch.isnan(x)) and torch.all(torch.isnan(y)):
        return 1.0

    if torch.all(torch.isinf(x)) and torch.all(torch.isinf(y)):
        if torch.all(x == y):
            return 1.0
        return 0.0

    mask = ~(torch.isnan(x) | torch.isinf(x) | torch.isnan(y) | torch.isinf(y))

    try:
        x = x[mask]
        y = y[mask]
    except RuntimeError:
        pass

    def equal(a, b, rtol=1e-2, atol=1e-2) -> float:
        return 1.0 if torch.allclose(a, b, rtol=rtol, atol=atol) else 0.0

    if min(x.numel(), y.numel()) < 2:
        return equal(x, y)

    x_centered = x - x.mean()
    y_centered = y - y.mean()

    sx2 = torch.sum(x_centered**2)
    sy2 = torch.sum(y_centered**2)
    if sx2.item() == 0.0 or sy2.item() == 0.0:
        return equal(x, y)

    numerator = torch.sum(x_centered * y_centered)
    denominator = torch.sqrt(sx2 * sy2)
    pcc = numerator / denominator

    return float(pcc)
```

- [ ] **Step 2: Update `tools/golden/CMakeLists.txt` — add `metrics.py` to sources**

```cmake
declare_mlir_python_sources(GoldenSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    __init__.py
    mapping.py
    metrics.py
)
```

- [ ] **Step 3: Update `tools/golden/__init__.py` — export metrics**

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from .mapping import *
from .metrics import compute_pcc, compute_atol, compute_rtol

__all__ = [
    "GoldenMapTensor",
    "unpack_mlir_attr",
    "get_golden_function",
    "GOLDEN_MAPPINGS",
    "compute_pcc",
    "compute_atol",
    "compute_rtol",
]
```

- [ ] **Step 4: Commit**

```bash
git add tools/golden/metrics.py tools/golden/CMakeLists.txt tools/golden/__init__.py
git commit -m "feat(golden): add unified metrics module (compute_pcc, compute_atol, compute_rtol)"
```

---

## Task 2: CMake Setup and Package Skeleton

**Files:**
- Create: `tools/chisel/CMakeLists.txt`
- Create: `tools/chisel/chisel/__init__.py` (empty initially)
- Modify: `tools/CMakeLists.txt`

- [ ] **Step 1: Create `tools/chisel/chisel/__init__.py`** (empty placeholder)

```python
# tools/chisel/chisel/__init__.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
```

- [ ] **Step 2: Create `tools/chisel/CMakeLists.txt`**

Follow the `tools/golden/CMakeLists.txt` pattern:

```cmake
include(AddMLIRPython)

declare_mlir_python_sources(ChiselSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    chisel/__init__.py
    chisel/ops.py
    chisel/executor.py
    chisel/context.py
    chisel/callbacks.py
    chisel/utils.py
)

add_mlir_python_modules(ChiselPythonModules
  ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/chisel"
  INSTALL_PREFIX "python_packages/chisel"
  DECLARED_SOURCES ChiselSources
)
```

- [ ] **Step 3: Add chisel to `tools/CMakeLists.txt`**

Add `add_subdirectory(chisel)` inside the Python bindings guard, after `golden`:

```cmake
if(TTMLIR_ENABLE_BINDINGS_PYTHON AND MLIR_ENABLE_BINDINGS_PYTHON)
  add_subdirectory(builder)
  add_subdirectory(golden)
  add_subdirectory(chisel)
  add_subdirectory(profiler)
  ...
```

- [ ] **Step 4: Commit**

```bash
git add tools/chisel/CMakeLists.txt tools/chisel/chisel/__init__.py tools/CMakeLists.txt
git commit -m "feat(chisel): add CMake package skeleton"
```

---

## Task 3: `utils.py` — Dtype Maps, Tensor Conversion, Debug Wrap

**Port from:** `runtime/tools/chisel/chisel/utils/runtime_utils.py` and `runtime/tools/chisel/chisel/utils/debug.py`

**Files:**
- Create: `tools/chisel/chisel/utils.py`
- Create: `tools/chisel/tests/test_utils.py`

- [ ] **Step 1: Write tests for dtype maps and debug_wrap**

```python
# tools/chisel/tests/test_utils.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch


def test_mlir_dtype_maps_contains_expected_keys():
    from chisel.utils import mlir_dtype_maps

    expected = {"i32", "f32", "bf16", "f16", "i1", "i64", "f64", "si32", "ui32"}
    assert expected.issubset(set(mlir_dtype_maps.keys()))


def test_mlir_dtype_maps_values_are_torch_dtypes():
    from chisel.utils import mlir_dtype_maps

    for key, dtype in mlir_dtype_maps.items():
        assert isinstance(dtype, torch.dtype), f"{key} maps to {dtype}, not a torch.dtype"


def test_ttrt_dtype_maps_values_are_torch_dtypes():
    from chisel.utils import ttrt_dtype_maps

    for key, dtype in ttrt_dtype_maps.items():
        assert isinstance(dtype, torch.dtype), f"{key} maps to {dtype}, not a torch.dtype"

```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tools/chisel/tests/test_utils.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'chisel.utils'`

- [ ] **Step 3: Create `tools/chisel/chisel/utils.py`**

```python
# tools/chisel/chisel/utils.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for Chisel: dtype maps, runtime tensor conversion, debug decorator.
"""
import functools

import torch
from ttrt.runtime import Tensor as RtTensor


mlir_dtype_maps = {
    "i32": torch.int32,
    "i64": torch.int64,
    "f32": torch.float32,
    "f64": torch.float64,
    "si32": torch.int32,
    "i1": torch.bool,
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "ui32": torch.uint32,
}


ttrt_dtype_maps = {
    "DataType.Float32": torch.float32,
    "DataType.BFloat16": torch.bfloat16,
    "DataType.UInt32": torch.uint32,
    "DataType.UInt16": torch.uint16,
    "DataType.UInt8": torch.uint8,
    "DataType.Int32": torch.int32,
}


def get_torch_tensor(tensor: RtTensor) -> torch.Tensor:
    """Convert a runtime tensor to a PyTorch tensor (copies data to host)."""
    rt_data_ptr = tensor.get_data_buffer()
    rt_dtype = tensor.get_dtype()
    dtype = ttrt_dtype_maps[str(rt_dtype)]
    shape = tensor.get_shape()
    torch_tensor = torch.frombuffer(rt_data_ptr, dtype=dtype)
    return torch_tensor.reshape(shape).clone()


def debug_wrap(*, debug: bool = False):
    """Decorator factory for runtime callbacks — drops into pdb on exception if debug=True."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception:
                if debug:
                    import pdb
                    import traceback

                    traceback.print_exc()
                    pdb.set_trace()
                raise

        return wrapper

    return decorator
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tools/chisel/tests/test_utils.py -v
```

Expected: All 6 tests PASS. Note: `get_torch_tensor` is not unit-tested here because it requires a live `RtTensor` — it is integration-tested via callbacks.

- [ ] **Step 5: Commit**

```bash
git add tools/chisel/chisel/utils.py tools/chisel/tests/test_utils.py
git commit -m "feat(chisel): add utils module — dtype maps, tensor conversion, debug_wrap"
```

---

## Task 4: `ops.py` — IRModule Wrapper and Op Extraction

**Port from:** `runtime/tools/chisel/chisel/core/ops.py`

**Changes from old version:**
- Remove `ExecutionType` parameter and attribute
- Remove `hash_location` / `_last_loc_line` (not needed for PR 1)
- Constructor accepts `mlir_source: str` and parses internally (instead of pre-parsed Module)
- Keep: `get_function()`, `get_function_inputs()`, `get_function_ops()`, `get_asm_state()`

**Files:**
- Create: `tools/chisel/chisel/ops.py`
- Create: `tools/chisel/tests/test_ops.py`

- [ ] **Step 1: Write tests for `get_op_inputs`, `get_op_outputs`, and `IRModule`**

```python
# tools/chisel/tests/test_ops.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest

# Minimal TTNN module for testing — uses generic MLIR tensor ops
# since ttnn dialect ops require registered dialects.
SIMPLE_MODULE = """
module {
  func.func @main(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttnn.add"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %1 = "ttnn.abs"(%0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }
}
"""


def test_ir_module_creation():
    from chisel.ops import IRModule

    ir = IRModule(mlir_source=SIMPLE_MODULE, functions=["main"])
    assert ir.module is not None
    assert ir.current_function_name == "main"


def test_get_function_returns_func_op():
    from chisel.ops import IRModule

    ir = IRModule(mlir_source=SIMPLE_MODULE, functions=["main"])
    func_op = ir.get_function()
    assert func_op.name.value == "main"


def test_get_function_inputs():
    from chisel.ops import IRModule

    ir = IRModule(mlir_source=SIMPLE_MODULE, functions=["main"])
    inputs = ir.get_function_inputs()
    assert len(inputs) == 2


def test_get_function_ops_returns_correct_count():
    from chisel.ops import IRModule

    ir = IRModule(mlir_source=SIMPLE_MODULE, functions=["main"])
    ops = ir.get_function_ops()
    # ttnn.add, ttnn.abs, func.return
    assert len(ops) == 3


def test_get_function_ops_order():
    from chisel.ops import IRModule

    ir = IRModule(mlir_source=SIMPLE_MODULE, functions=["main"])
    ops = ir.get_function_ops()
    assert ops[0].name == "ttnn.add"
    assert ops[1].name == "ttnn.abs"
    assert ops[2].name == "func.return"


def test_ignored_ops():
    from chisel.ops import IRModule

    ir = IRModule(
        mlir_source=SIMPLE_MODULE,
        functions=["main"],
        ignored_ops=["func.return"],
    )
    ops = ir.get_function_ops()
    assert len(ops) == 2
    assert all(op.name != "func.return" for op in ops)


def test_get_asm_state():
    from chisel.ops import IRModule

    ir = IRModule(mlir_source=SIMPLE_MODULE, functions=["main"])
    asm_state = ir.get_asm_state()
    assert asm_state is not None


def test_get_op_inputs():
    from chisel.ops import IRModule, get_op_inputs

    ir = IRModule(mlir_source=SIMPLE_MODULE, functions=["main"])
    add_op = ir.get_function_ops()[0]  # ttnn.add
    inputs = get_op_inputs(add_op)
    assert len(inputs) == 2


def test_get_op_outputs():
    from chisel.ops import IRModule, get_op_outputs

    ir = IRModule(mlir_source=SIMPLE_MODULE, functions=["main"])
    add_op = ir.get_function_ops()[0]  # ttnn.add
    outputs = get_op_outputs(add_op)
    assert len(outputs) == 1


def test_get_op_inputs_for_unary():
    from chisel.ops import IRModule, get_op_inputs

    ir = IRModule(mlir_source=SIMPLE_MODULE, functions=["main"])
    abs_op = ir.get_function_ops()[1]  # ttnn.abs
    inputs = get_op_inputs(abs_op)
    assert len(inputs) == 1


def test_missing_function_raises():
    from chisel.ops import IRModule

    with pytest.raises(ValueError, match="not found"):
        IRModule(mlir_source=SIMPLE_MODULE, functions=["nonexistent"])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tools/chisel/tests/test_ops.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'chisel.ops'`

- [ ] **Step 3: Create `tools/chisel/chisel/ops.py`**

```python
# tools/chisel/chisel/ops.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MLIR operation utilities: IRModule wrapper and tensor operand extraction.
"""
from functools import cache
from typing import Dict, List

from ttmlir.dialects import func
from ttmlir.ir import (
    AsmState,
    Context,
    Module,
    Operation,
    WalkOrder,
    WalkResult,
    BlockArgument,
)


@cache
def get_op_outputs(op: Operation) -> list:
    """Extract output tensors (results with shape and element_type) from an MLIR operation."""
    outputs = []
    for result in op.results:
        if hasattr(result.type, "shape") and hasattr(result.type, "element_type"):
            outputs.append(result)
    return outputs


@cache
def get_op_inputs(op: Operation) -> list:
    """Extract input tensors (operands with shape and element_type) from an MLIR operation."""
    inputs = []
    for operand in op.operands:
        if hasattr(operand.type, "shape") and hasattr(operand.type, "element_type"):
            inputs.append(operand)
    return inputs


class IRModule:
    """
    Wrapper around an MLIR Module with function lookup and operation traversal.

    Accepts an MLIR source string, parses it internally, and provides cached
    access to functions, operations, and assembly state.
    """

    def __init__(
        self,
        mlir_source: str,
        functions: List[str],
        current_function_name: str | None = None,
        ignored_ops: List[str] = [],
    ):
        self.context = Context()
        self.context.allow_unregistered_dialects = True
        self.module: Module = Module.parse(mlir_source, self.context)
        self.ignored_ops: List[str] = ignored_ops

        self._functions: Dict[str, Operation] = {
            name: self._find_function(name) for name in functions
        }
        self._function_ops: Dict[str, List[Operation]] = {
            name: self._extract_function_ops(name) for name in functions
        }
        self._asm_state: Dict[str, AsmState] = {
            name: AsmState(self._functions[name]) for name in functions
        }

        if current_function_name is not None:
            self.current_function_name = current_function_name
        else:
            self.current_function_name = functions[0]

    def get_asm_state(self) -> AsmState:
        """AsmState for the current function (speeds up get_name calls)."""
        return self._asm_state[self.current_function_name]

    def get_function(self) -> Operation:
        """The current func.FuncOp."""
        return self._functions[self.current_function_name]

    def get_function_inputs(self) -> List[BlockArgument]:
        """Input arguments of the current function."""
        return self._functions[self.current_function_name].arguments

    def get_function_ops(self) -> List[Operation]:
        """Operations in the current function body (respecting ignored_ops)."""
        return self._function_ops[self.current_function_name]

    def _extract_function_ops(self, name: str) -> List[Operation]:
        assert name in self._functions
        ops = []
        for region in self._functions[name].regions:
            for block in region.blocks:
                for op in block.operations:
                    if op.name in self.ignored_ops:
                        continue
                    ops.append(op)
        return ops

    def _find_function(self, name: str) -> Operation:
        for op in self._walk(self.module.operation):
            if isinstance(op, func.FuncOp) and op.name.value == name:
                return op
        raise ValueError(f"Function {name} not found in module")

    def _walk(
        self, op: Operation, walk_order: WalkOrder = WalkOrder.POST_ORDER
    ) -> List[Operation]:
        ops = []

        def _walk_ops(op):
            nonlocal ops
            ops.append(op.opview)
            return WalkResult.ADVANCE

        op.operation.walk(_walk_ops, walk_order=walk_order)
        return ops
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tools/chisel/tests/test_ops.py -v
```

Expected: All 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/chisel/chisel/ops.py tools/chisel/tests/test_ops.py
git commit -m "feat(chisel): add ops module — IRModule wrapper and op extraction"
```

---

## Task 5: `executor.py` — Golden Execution via Direct Dispatch

**Simple approach:** Golden functions accept `*operands, **attributes` directly.
We call them as:
```python
golden_fn(
    *[GoldenMapTensor({0: inputs[o.get_name(asm)]}, (1,1)) for o in op.operands],
    **{name: value for name, value in op.attributes}
)
```

No signature introspection needed — MLIR attributes are passed as keyword
arguments and golden functions accept what they need via `**kwargs`.

**Files:**
- Create: `tools/chisel/chisel/executor.py`
- Create: `tools/chisel/tests/test_executor.py`

- [ ] **Step 1: Write tests for `execute_golden`**

```python
# tools/chisel/tests/test_executor.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

UNARY_MODULE = """
module {
  func.func @main(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = "ttnn.abs"(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
"""

BINARY_MODULE = """
module {
  func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = "ttnn.add"(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
"""


def test_execute_golden_abs():
    """End-to-end test: execute golden abs on a tensor via MLIR op."""
    from chisel.ops import IRModule, get_op_inputs
    from chisel.executor import execute_golden

    ir = IRModule(mlir_source=UNARY_MODULE, functions=["main"])
    abs_op = ir.get_function_ops()[0]  # ttnn.abs

    # Build inputs dict keyed by SSA name
    op_inputs = get_op_inputs(abs_op)
    input_name = op_inputs[0].get_name(ir.get_asm_state())
    input_tensor = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
    inputs = {input_name: input_tensor}

    result = execute_golden(abs_op, ir, inputs)
    expected = torch.abs(input_tensor)
    assert torch.allclose(result, expected)


def test_execute_golden_add():
    """End-to-end test: execute golden add on two tensors."""
    from chisel.ops import IRModule, get_op_inputs
    from chisel.executor import execute_golden

    ir = IRModule(mlir_source=BINARY_MODULE, functions=["main"])
    add_op = ir.get_function_ops()[0]  # ttnn.add

    op_inputs = get_op_inputs(add_op)
    name0 = op_inputs[0].get_name(ir.get_asm_state())
    name1 = op_inputs[1].get_name(ir.get_asm_state())
    t0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    t1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    inputs = {name0: t0, name1: t1}

    result = execute_golden(add_op, ir, inputs)
    expected = torch.add(t0, t1)
    assert torch.allclose(result, expected)


def test_execute_golden_unmapped_op_raises():
    """Verify RuntimeError for ops not in GOLDEN_MAPPINGS."""
    from chisel.ops import IRModule
    from chisel.executor import execute_golden

    # func.return is not in GOLDEN_MAPPINGS
    ir = IRModule(mlir_source=UNARY_MODULE, functions=["main"])
    return_op = ir.get_function_ops()[-1]  # func.return

    with pytest.raises(RuntimeError, match="No golden implementation"):
        execute_golden(return_op, ir, {})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tools/chisel/tests/test_executor.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'chisel.executor'`

- [ ] **Step 3: Create `tools/chisel/chisel/executor.py`**

```python
# tools/chisel/chisel/executor.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Golden execution for a single TTNN op via GOLDEN_MAPPINGS.

Calls golden functions with positional operand tensors and keyword MLIR
attributes directly — no signature introspection needed.
"""
import torch
from ttmlir.ir import Operation

from golden import get_golden_function, GoldenMapTensor

from chisel.ops import IRModule, get_op_inputs, get_op_outputs


def execute_golden(op: Operation, ir_module: IRModule, inputs: dict) -> torch.Tensor:
    """
    Execute a TTNN op on CPU via GOLDEN_MAPPINGS.

    Args:
        op: The MLIR operation to execute.
        ir_module: The IRModule containing the operation (for SSA name resolution).
        inputs: Dict mapping SSA names to torch.Tensor (device inputs copied to host).

    Returns:
        torch.Tensor — the golden output.

    Raises:
        RuntimeError: If no golden implementation exists for the op type.
    """
    golden_fn = get_golden_function(type(op))
    if golden_fn is None:
        raise RuntimeError(f"No golden implementation for {type(op).__name__}")

    # Wrap input operand tensors as GoldenMapTensor (single-device)
    asm_state = ir_module.get_asm_state()
    golden_inputs = [
        GoldenMapTensor({0: inputs[operand.get_name(asm_state)]}, (1, 1))
        for operand in get_op_inputs(op)
    ]

    # Pass all MLIR attributes as keyword arguments
    attrs = {name: value for name, value in op.attributes}

    # Call golden function: positional operands + keyword attributes
    result = golden_fn(*golden_inputs, **attrs)

    # Extract torch.Tensor from GoldenMapTensor
    if isinstance(result, GoldenMapTensor):
        return result.golden_map_tensor_as_torch_tensors()[0]
    if isinstance(result, torch.Tensor):
        return result
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tools/chisel/tests/test_executor.py -v
```

Expected: All 3 tests PASS. Note: `test_execute_golden_abs` and `test_execute_golden_add`
require TTNN ops to be in `GOLDEN_MAPPINGS`. If generic MLIR ops are used (unregistered
dialect), `type(op)` won't match `ttnn.AbsOp` — these tests need `ttmlir` bindings with
TTNN dialect registered. If this is an issue, gate with
`pytest.importorskip("ttmlir.dialects.ttnn")` and adjust test module strings.

- [ ] **Step 5: Commit**

```bash
git add tools/chisel/chisel/executor.py tools/chisel/tests/test_executor.py
git commit -m "feat(chisel): add executor — golden execution via direct dispatch"
```

---

## Task 6: `context.py` — Slim ChiselContext Singleton

**Files:**
- Create: `tools/chisel/chisel/context.py`
- Create: `tools/chisel/tests/test_context.py`

- [ ] **Step 1: Write tests for ChiselContext singleton lifecycle**

```python
# tools/chisel/tests/test_context.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest

SIMPLE_MODULE = """
module {
  func.func @main(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = "ttnn.abs"(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %1 = "ttnn.neg"(%0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}
"""


@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensure singleton is reset before and after each test."""
    from chisel.context import ChiselContext

    ChiselContext.reset_instance()
    yield
    ChiselContext.reset_instance()


def test_get_instance_before_init_raises():
    from chisel.context import ChiselContext

    with pytest.raises(RuntimeError, match="not initialized"):
        ChiselContext.get_instance()


def test_construction_sets_instance():
    from chisel.context import ChiselContext
    from chisel.ops import IRModule

    ir = IRModule(mlir_source=SIMPLE_MODULE, functions=["main"])
    ctx = ChiselContext(ir_module=ir)
    assert ChiselContext.get_instance() is ctx


def test_singleton_returns_same_object():
    from chisel.context import ChiselContext
    from chisel.ops import IRModule

    ir = IRModule(mlir_source=SIMPLE_MODULE, functions=["main"])
    ctx = ChiselContext(ir_module=ir)
    assert ChiselContext.get_instance() is ctx
    assert ChiselContext.get_instance() is ctx


def test_reset_clears_instance():
    from chisel.context import ChiselContext
    from chisel.ops import IRModule

    ir = IRModule(mlir_source=SIMPLE_MODULE, functions=["main"])
    ChiselContext(ir_module=ir)
    ChiselContext.reset_instance()

    with pytest.raises(RuntimeError, match="not initialized"):
        ChiselContext.get_instance()


def test_op_iter_advances():
    from chisel.context import ChiselContext
    from chisel.ops import IRModule

    ir = IRModule(
        mlir_source=SIMPLE_MODULE,
        functions=["main"],
        ignored_ops=["func.return"],
    )
    ctx = ChiselContext(ir_module=ir)

    op1 = next(ctx.op_iter)
    assert op1.name == "ttnn.abs"

    op2 = next(ctx.op_iter)
    assert op2.name == "ttnn.neg"


def test_op_iter_exhaustion():
    from chisel.context import ChiselContext
    from chisel.ops import IRModule

    ir = IRModule(
        mlir_source=SIMPLE_MODULE,
        functions=["main"],
        ignored_ops=["func.return"],
    )
    ctx = ChiselContext(ir_module=ir)
    next(ctx.op_iter)  # abs
    next(ctx.op_iter)  # neg

    with pytest.raises(StopIteration):
        next(ctx.op_iter)


def test_stashed_inputs_lifecycle():
    from chisel.context import ChiselContext
    from chisel.ops import IRModule

    ir = IRModule(mlir_source=SIMPLE_MODULE, functions=["main"])
    ctx = ChiselContext(ir_module=ir)

    assert ctx._stashed_inputs is None

    ctx._stashed_inputs = {"arg0": "tensor_data"}
    assert ctx._stashed_inputs["arg0"] == "tensor_data"

    ctx._stashed_inputs = None
    assert ctx._stashed_inputs is None


def test_current_op_lifecycle():
    from chisel.context import ChiselContext
    from chisel.ops import IRModule

    ir = IRModule(mlir_source=SIMPLE_MODULE, functions=["main"])
    ctx = ChiselContext(ir_module=ir)

    assert ctx._current_op is None
    ctx._current_op = "mock_op"
    assert ctx._current_op == "mock_op"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tools/chisel/tests/test_context.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'chisel.context'`

- [ ] **Step 3: Create `tools/chisel/chisel/context.py`**

```python
# tools/chisel/chisel/context.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Slim ChiselContext singleton for per-op isolation testing.

Holds only what's needed for PR 1: ir_module, op_iter, stashed inputs.
No BinaryState/ProgramState hierarchy (added in PR 2).
"""
from typing import Iterator, Optional

from ttmlir.ir import Operation

from chisel.ops import IRModule


class ChiselContext:
    """Singleton context for chisel callbacks during TTRT execution."""

    _instance: Optional["ChiselContext"] = None

    def __init__(self, ir_module: IRModule):
        ChiselContext._instance = self
        self.ir_module = ir_module
        self.op_iter: Iterator = iter(ir_module.get_function_ops())
        self._current_op: Operation | None = None
        self._stashed_inputs: dict | None = None

    @classmethod
    def get_instance(cls) -> "ChiselContext":
        if cls._instance is None:
            raise RuntimeError("ChiselContext not initialized")
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tools/chisel/tests/test_context.py -v
```

Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/chisel/chisel/context.py tools/chisel/tests/test_context.py
git commit -m "feat(chisel): add slim ChiselContext singleton"
```

---

## Task 7: `callbacks.py` — preOp / postOp Hooks

**Files:**
- Create: `tools/chisel/chisel/callbacks.py`
- Create: `tools/chisel/tests/test_callbacks.py`

- [ ] **Step 1: Write tests for callbacks**

These tests mock the runtime objects since they require hardware.

```python
# tools/chisel/tests/test_callbacks.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from unittest.mock import MagicMock, patch

SIMPLE_MODULE = """
module {
  func.func @main(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = "ttnn.abs"(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
"""


@pytest.fixture(autouse=True)
def reset_singleton():
    from chisel.context import ChiselContext

    ChiselContext.reset_instance()
    yield
    ChiselContext.reset_instance()


@pytest.fixture
def ctx_with_module():
    from chisel.context import ChiselContext
    from chisel.ops import IRModule

    ir = IRModule(
        mlir_source=SIMPLE_MODULE,
        functions=["main"],
        ignored_ops=["func.return"],
    )
    return ChiselContext(ir_module=ir)


def test_pre_op_advances_op_iter(ctx_with_module):
    from chisel.callbacks import chisel_pre_op_callback

    mock_binary = MagicMock()
    mock_program_ctx = MagicMock()
    mock_op_ctx = MagicMock()

    # Mock runtime functions to return empty refs (no inputs to stash)
    with patch("chisel.callbacks.get_op_input_refs", return_value=[]):
        chisel_pre_op_callback(mock_binary, mock_program_ctx, mock_op_ctx)

    ctx = ctx_with_module
    assert ctx._current_op is not None
    assert ctx._current_op.name == "ttnn.abs"


def test_pre_op_stashes_inputs(ctx_with_module):
    from chisel.callbacks import chisel_pre_op_callback

    mock_binary = MagicMock()
    mock_program_ctx = MagicMock()
    mock_op_ctx = MagicMock()

    # Create a mock runtime tensor
    mock_rt_tensor = MagicMock()
    mock_rt_tensor.get_data_buffer.return_value = torch.randn(4, 4).numpy().data
    mock_rt_tensor.get_dtype.return_value = "Float32"
    mock_rt_tensor.get_shape.return_value = [4, 4]

    mock_ref = MagicMock()

    with (
        patch("chisel.callbacks.get_op_input_refs", return_value=[mock_ref]),
        patch("chisel.callbacks.retrieve_tensor_from_pool", return_value=mock_rt_tensor),
        patch("chisel.callbacks.get_torch_tensor", return_value=torch.randn(4, 4)),
    ):
        chisel_pre_op_callback(mock_binary, mock_program_ctx, mock_op_ctx)

    ctx = ctx_with_module
    assert ctx._stashed_inputs is not None
    assert len(ctx._stashed_inputs) == 1


def test_post_op_clears_stash(ctx_with_module):
    from chisel.callbacks import chisel_pre_op_callback, chisel_post_op_callback

    mock_binary = MagicMock()
    mock_program_ctx = MagicMock()
    mock_op_ctx = MagicMock()

    input_tensor = torch.randn(4, 4)
    golden_output = torch.abs(input_tensor)
    device_output = torch.abs(input_tensor) + 0.001  # slight diff

    mock_ref = MagicMock()

    # Pre-op: stash inputs
    with (
        patch("chisel.callbacks.get_op_input_refs", return_value=[mock_ref]),
        patch("chisel.callbacks.retrieve_tensor_from_pool", return_value=MagicMock()),
        patch("chisel.callbacks.get_torch_tensor", return_value=input_tensor),
    ):
        chisel_pre_op_callback(mock_binary, mock_program_ctx, mock_op_ctx)

    # Post-op: execute golden and compare
    with (
        patch("chisel.callbacks.execute_golden", return_value=golden_output),
        patch("chisel.callbacks.get_op_output_ref", return_value=mock_ref),
        patch("chisel.callbacks.retrieve_tensor_from_pool", return_value=MagicMock()),
        patch("chisel.callbacks.get_torch_tensor", return_value=device_output),
    ):
        chisel_post_op_callback(mock_binary, mock_program_ctx, mock_op_ctx)

    ctx = ctx_with_module
    assert ctx._stashed_inputs is None


def test_callback_without_context_raises():
    from chisel.callbacks import chisel_pre_op_callback

    with pytest.raises(RuntimeError, match="not initialized"):
        chisel_pre_op_callback(MagicMock(), MagicMock(), MagicMock())
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tools/chisel/tests/test_callbacks.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'chisel.callbacks'`

- [ ] **Step 3: Create `tools/chisel/chisel/callbacks.py`**

```python
# tools/chisel/chisel/callbacks.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DebugHooks callbacks for per-op isolation testing.

Two plain functions compatible with DebugHooks op-level callbacks.
No program-level callbacks in this PR.
"""
import logging

from ttrt.runtime import (
    get_op_input_refs,
    get_op_output_ref,
    retrieve_tensor_from_pool,
)

from chisel.context import ChiselContext
from chisel.ops import get_op_inputs, get_op_outputs
from chisel.executor import execute_golden
from chisel.utils import get_torch_tensor
from golden.metrics import compute_pcc, compute_atol, compute_rtol

logger = logging.getLogger("chisel")


def chisel_pre_op_callback(binary, program_context, op_context):
    """
    Pre-operation callback: advance op iterator and stash device input tensors.

    1. Advance op_iter to get current MLIR op
    2. Copy device input tensors to host
    3. Stash inputs in ctx._stashed_inputs for postOp
    """
    ctx = ChiselContext.get_instance()
    ctx._current_op = next(ctx.op_iter)

    # Copy device inputs to host, keyed by SSA name
    ctx._stashed_inputs = {}
    op_inputs = get_op_inputs(ctx._current_op)
    input_refs = get_op_input_refs(op_context, program_context)
    asm_state = ctx.ir_module.get_asm_state()

    for mlir_input, tensor_ref in zip(op_inputs, input_refs):
        device_tensor = retrieve_tensor_from_pool(program_context, tensor_ref)
        name = mlir_input.get_name(asm_state)
        ctx._stashed_inputs[name] = get_torch_tensor(device_tensor)


def chisel_post_op_callback(binary, program_context, op_context):
    """
    Post-operation callback: run golden, capture device output, compare, log.

    1. Run golden function with stashed inputs
    2. Capture device output tensor
    3. Compare golden vs device (PCC, atol, rtol)
    4. Log metrics
    5. Discard golden output (no pool storage)
    """
    ctx = ChiselContext.get_instance()

    # Skip ops with no outputs
    if len(get_op_outputs(ctx._current_op)) == 0:
        ctx._stashed_inputs = None
        return

    # Execute golden
    golden_result = execute_golden(
        ctx._current_op, ctx.ir_module, ctx._stashed_inputs
    )

    # Capture device output
    output_ref = get_op_output_ref(op_context, program_context)
    device_tensor = retrieve_tensor_from_pool(program_context, output_ref)
    device_torch = get_torch_tensor(device_tensor)

    # Compare
    pcc = compute_pcc(golden_result, device_torch)
    atol = compute_atol(golden_result, device_torch)
    rtol = compute_rtol(golden_result, device_torch)

    # Log
    op_name = ctx._current_op.name
    logger.info(f"{op_name}: PCC={pcc:.6f}, atol={atol:.6e}, rtol={rtol:.6e}")

    # Discard — no pool storage in isolation mode
    ctx._stashed_inputs = None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tools/chisel/tests/test_callbacks.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/chisel/chisel/callbacks.py tools/chisel/tests/test_callbacks.py
git commit -m "feat(chisel): add preOp/postOp callbacks for isolation testing"
```

---

## Task 8: Package Exports and Final Wiring

**Files:**
- Modify: `tools/chisel/chisel/__init__.py`

- [ ] **Step 1: Update `tools/chisel/chisel/__init__.py` with public exports**

```python
# tools/chisel/chisel/__init__.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from chisel.context import ChiselContext
from chisel.callbacks import (
    chisel_pre_op_callback,
    chisel_post_op_callback,
)
```

- [ ] **Step 2: Verify all tests pass together**

```bash
pytest tools/chisel/tests/ -v
```

Expected: All tests across all modules pass.

- [ ] **Step 3: Commit**

```bash
git add tools/chisel/chisel/__init__.py
git commit -m "feat(chisel): add package exports"
```

---

## Task 9: Update PR 1 Doc — Resolve TODO and Update Dependencies

**Files:**
- Modify: `tools/chisel/docs/pr1_single_op_isolation.md`

- [ ] **Step 1: Remove the TODO comment and update the Metrics section**

In `tools/chisel/docs/pr1_single_op_isolation.md`, replace the TODO comment block (around line 256) with updated text reflecting that `golden/metrics.py` is created as part of this PR:

```markdown
### Metrics — `tools/golden/metrics.py`

Created as part of this PR. Pure-torch implementations of `compute_pcc`,
`compute_atol`, `compute_rtol` ported from the old chisel metrics at
`runtime/tools/chisel/chisel/utils/metrics.py`. This removes the PR 0c
dependency.

```python
from golden.metrics import compute_pcc, compute_atol, compute_rtol
```

- [ ] **Step 2: Update the Dependencies section**

Remove PR 0c from the dependency list since metrics are now included:

```markdown
## Dependencies

- **PR 0a-1** — GIL-Safety Fix (callbacks must not be copied)
- **PR 0a-2a** — Named Callback API (register preOp/postOp by name)

Does **not** require:
- PR 0a-2b (program-level hooks) — no preProgram/postProgram in this PR
- PR 0a-3 (introspection bindings) — no program_index or input_refs queries
- PR 0c (unified metrics) — `golden/metrics.py` is created in this PR
```

- [ ] **Step 3: Commit**

```bash
git add tools/chisel/docs/pr1_single_op_isolation.md
git commit -m "docs(chisel): resolve metrics TODO, update PR 1 dependencies"
```

---

## Notes for the Implementer

### Test MLIR Modules and Dialect Registration

The test MLIR modules use generic op syntax (`"ttnn.abs"(...)` with quotes) which
works with `context.allow_unregistered_dialects = True`. This means `type(op)` returns
a generic `Operation` rather than `ttnn.AbsOp`. For `execute_golden` tests to work
end-to-end with `GOLDEN_MAPPINGS`, the ttmlir Python bindings must be built and the
TTNN dialect registered. If tests are run without the full build, only the
`test_execute_golden_unmapped_op_raises` test will pass (the others need dialect
registration to match `type(op)` in GOLDEN_MAPPINGS).

If dialect registration is needed for integration tests, register dialects in the
`IRModule` constructor:

```python
from ttmlir.dialects import ttnn  # registers the dialect
```

### Golden Function Calling Convention

Golden functions are called with direct dispatch:
```python
golden_fn(*golden_inputs, **{name: value for name, value in op.attributes})
```

This works because golden functions accept their operands as positional args and
MLIR attributes as keyword args. The MLIR attribute names match the golden function
parameter names. If a golden function doesn't use a particular attribute, Python's
`**kwargs` silently absorbs it.

### Running Tests

```bash
# Unit tests (no hardware needed, needs ttmlir Python bindings):
source env/activate
pytest tools/chisel/tests/ -v

# Specific module:
pytest tools/chisel/tests/test_executor.py -v
```

### Key API Reference

- `get_golden_function(type(op))` → returns callable or asserts (from `tools/golden/mapping.py:7154`)
- `GoldenMapTensor({0: tensor}, (1, 1))` → single-device wrapper (from `tools/golden/mapping.py:27`)
- `result.golden_map_tensor_as_torch_tensors()[0]` → extract torch.Tensor from device 0
- `get_op_input_refs(op_context, program_context)` → list of `TensorRef` (from `ttrt.runtime`)
- `get_op_output_ref(op_context, program_context)` → single `TensorRef`
- `retrieve_tensor_from_pool(program_context, ref)` → `RtTensor`
- `value.get_name(asm_state)` → SSA name string like `%0`, `%arg0`
