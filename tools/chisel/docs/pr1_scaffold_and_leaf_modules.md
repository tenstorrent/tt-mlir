# PR 1: Scaffold + Data Structures

## Goal

Establish the importable `chisel` Python package with CMake packaging, plus the
core data structures that hold state: `TensorPool` for tensor management and
`IRModule` for MLIR module traversal. This gives a testable
foundation from the first PR.

## Files

### New Files

| File | Description |
|------|-------------|
| `tools/chisel/CMakeLists.txt` | CMake packaging using `declare_mlir_python_sources` |
| `tools/chisel/chisel/__init__.py` | Minimal package init |
| `tools/chisel/chisel/tensors.py` | `TensorPool` (stores `GoldenMapTensor` directly) |
| `tools/chisel/chisel/ops.py` | `IRModule` wrapper, `get_op_inputs()`, `get_op_outputs()` |

### Modified Files

| File | Change |
|------|--------|
| `tools/CMakeLists.txt` | Add `add_subdirectory(chisel)` under the Python bindings guard |

## Implementation Details

### `CMakeLists.txt`

Follow the `tools/builder/CMakeLists.txt` and `tools/golden/CMakeLists.txt` pattern
(`declare_mlir_python_sources` to register files, then `add_mlir_python_modules`
to copy them into the build/install tree so the package is importable):

```cmake
include(AddMLIRPython)

declare_mlir_python_sources(ChiselSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    chisel/__init__.py
    chisel/tensors.py
    chisel/ops.py
)

add_mlir_python_modules(ChiselPythonModules
  ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/chisel"
  INSTALL_PREFIX "python_packages/chisel"
  DECLARED_SOURCES ChiselSources
)
```

Add to `tools/CMakeLists.txt`:
```cmake
if(TTMLIR_ENABLE_BINDINGS_PYTHON AND MLIR_ENABLE_BINDINGS_PYTHON)
  add_subdirectory(builder)
  add_subdirectory(golden)
  add_subdirectory(chisel)   # <-- new
endif()
```

### `ops.py`

**Utility functions:**

```python
@cache
def get_op_outputs(op: Operation) -> list:
    """Extract tensor-like outputs (results with shape and element_type)."""

@cache
def get_op_inputs(op: Operation) -> list:
    """Extract tensor-like inputs (operands with shape and element_type)."""
```

**`IRModule`** — wraps an MLIR Module with caching and traversal. Accepts an
MLIR source string and parses it internally:

```python
class IRModule:
    def __init__(
        self,
        mlir_source: str,
        functions: List[str],
        current_function_name: str | None = None,
        ignored_ops: List[str] = [],
    ):
        # Parse the MLIR source string into a Module
        self.context = Context()
        self.module = Module.parse(mlir_source, self.context)
        ...

    def get_function(self) -> Operation: ...
    def get_function_inputs(self) -> List[BlockArgument]: ...
    def get_function_ops(self) -> List[Operation]: ...
    def get_asm_state(self) -> AsmState: ...

```

### `tensors.py`

**`TensorPool`** — dict subclass mapping keys to `GoldenMapTensor` directly,
with optional disk caching. No `TensorValue` or `DeviceHandle` wrapper needed:

- `GoldenMapTensor` (from `tools/golden/mapping.py`) already provides all
  needed tensor semantics (sharding, `__torch_function__` for shard-wise ops,
  dtype conversion via `golden_map_tensor_as_torch_tensors()`)
- Device tensors are ephemeral — captured from the runtime API in each callback
  and consumed immediately, so they don't need pool storage or a `DeviceHandle`
- The old `TensorValue.snapshot`/`working` split is unnecessary because golden
  ops don't mutate inputs (they return new `GoldenMapTensor` instances)

```python
class TensorPool(dict):
    """Dict mapping SSA name (or globalId) -> GoldenMapTensor, with optional disk caching."""

    def __init__(self, caching: bool = False, output_dir: Path | None = None):
        super().__init__()
        self.caching = caching
        self.output_dir = output_dir
        if caching and output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

    def __setitem__(self, key, value: GoldenMapTensor):
        super().__setitem__(key, value)
        if not self.caching:
            return
        torch_tensors = value.golden_map_tensor_as_torch_tensors()
        for device_id, tensor in torch_tensors.items():
            if tensor.dtype in [torch.uint16, torch.uint32, torch.uint64]:
                continue
            torch.save(tensor, self.output_dir / f"{key}_dev{device_id}.pt")
```

## Porting Notes

### `ops.py` from `runtime/tools/chisel/chisel/core/ops.py`

**`get_op_outputs()` and `get_op_inputs()`:**
- **Port as-is** — pure MLIR operation introspection, no ExecutionType dependency

**`IRModule` changes:**
- **Remove** `execution_type: ExecutionType` constructor parameter and attribute
- **Remove** `self.execution_type` — used only in `__repr__` and passed to AsmState
- **Keep as-is:** `get_function()`, `get_function_inputs()`, `get_function_ops()`,
  `get_asm_state()`
- The `ignored_ops` parameter stays — useful for skipping `ttnn.deallocate` and
  similar non-compute ops

### `tensors.py` from `runtime/tools/chisel/chisel/core/tensors.py`

**Simplified — drop `TensorValue` and `DeviceHandle` entirely:**
- **Remove** `TensorValue` class — pool stores `GoldenMapTensor` directly
- **Remove** `DeviceHandle` — device tensor read/write stays inline in callbacks
- **Keep** `TensorPool(dict)` with optional disk caching
- **Update** caching to use `GoldenMapTensor.golden_map_tensor_as_torch_tensors()`
  for serialization (saves per-shard `.pt` files)
- In the new design, `golden_tensor_pool` (CPU tensors) is the only
  `TensorPool` on `ProgramState`. Device tensors are ephemeral — captured from
  the runtime API in each callback and consumed immediately.

## Test Plan

### `test_tensors.py`
- `test_tensor_pool_insert_retrieve()` — insert `GoldenMapTensor`, retrieve by key
- `test_tensor_pool_disk_caching()` — create pool with `caching=True` and tmp dir,
  insert `GoldenMapTensor`, verify per-shard `.pt` files written to disk
- `test_tensor_pool_is_dict()` — verify TensorPool behaves as dict (keys, values, items, len)
- `test_tensor_pool_no_caching()` — verify no files written when `caching=False`

**Test dependencies:** `torch` only.

### `test_ops.py`
- `test_ir_module_creation()` — parse a small TTNN MLIR module string, create IRModule
- `test_get_function()` — verify `get_function()` returns the expected function op
- `test_get_function_ops()` — verify operations are listed in correct order
- `test_get_op_inputs_outputs()` — verify tensor-like operand/result extraction
- `test_ignored_ops()` — verify ops in `ignored_ops` list are filtered

**Test dependencies:** `ttmlir` Python bindings for MLIR module parsing. Tests
parse small inline MLIR strings — no hardware needed.

**Example test fixture:**
```python
SIMPLE_TTNN_MODULE = """
module {
  func.func @main(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttnn.abs"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
"""
```

## Dependencies

None — this is the first PR in the chain.
