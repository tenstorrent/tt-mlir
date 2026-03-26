# PR 1: Scaffold + Data Structures

## Goal

Establish the importable `chisel` Python package with CMake packaging, plus the
core data structures that hold state: `TensorPool`/`TensorValue` for tensor
management and `IRModule` for MLIR module traversal. This gives a testable
foundation from the first PR.

## Files

### New Files

| File | Description |
|------|-------------|
| `tools/chisel/CMakeLists.txt` | CMake packaging using `declare_mlir_python_sources` |
| `tools/chisel/chisel/__init__.py` | Minimal package init |
| `tools/chisel/chisel/tensors.py` | `DeviceHandle`, `TensorValue`, and `TensorPool` |
| `tools/chisel/chisel/ops.py` | `IRModule` wrapper, `get_op_inputs()`, `get_op_outputs()`, `hash_location()` |

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

**Location utilities** (inlined, no separate utils.py needed yet):

```python
from typing import Tuple
from ttmlir.ir import Location

UNKNOWN_LOCATION = (-1, -1)

def hash_location(location: Location) -> Tuple[int, int]:
    assert location is not None
    if not hasattr(location, "start_line"):
        return UNKNOWN_LOCATION
    if not hasattr(location, "start_col"):
        return UNKNOWN_LOCATION
    return (location.start_line, location.start_col)
```

These are inlined into `ops.py` because they are tiny (5 lines) and only used
here and in `registry.py` (PR 2), which imports from `ops.py`. This avoids
creating a `utils.py` module that can't be meaningfully tested in isolation.

**Utility functions:**

```python
@cache
def get_op_outputs(op: Operation) -> list:
    """Extract tensor-like outputs (results with shape and element_type)."""

@cache
def get_op_inputs(op: Operation) -> list:
    """Extract tensor-like inputs (operands with shape and element_type)."""
```

**`IRModule`** — wraps an MLIR Module with caching and traversal:

```python
class IRModule:
    def __init__(
        self,
        mlir_module: Module,
        context: Context,
        functions: List[str],
        current_function_name: str | None = None,
        ignored_ops: List[str] = [],
    ):
        ...

    def get_function(self) -> Operation: ...
    def get_function_inputs(self) -> List[BlockArgument]: ...
    def get_function_ops(self) -> List[Operation]: ...
    def get_asm_state(self) -> AsmState: ...

    @property
    def last_loc_line(self) -> Dict[Tuple[int, int], int]:
        """Map of (line, column) locations to operation indices."""
```

### `tensors.py`

**`DeviceHandle`** — dataclass encapsulating device runtime pool interaction:

```python
@dataclass
class DeviceHandle:
    """Encapsulates device runtime pool interaction for a single tensor."""
    tensor_ref: TensorRef
    _cached: Tensor | None = field(default=None, repr=False)

    def write_to_pool(self, program_context, data): ...
    def read_from_pool(self, program_context) -> torch.Tensor | None: ...
```

**`TensorValue`** — a named tensor with a snapshot for comparison and a working
value for execution. Uses composition with `DeviceHandle` for device tensors:

```python
class TensorValue:
    """
    For golden (CPU) tensors: device is None, working holds GoldenMapTensor.
    For device (HW) tensors: device holds DeviceHandle, working is staging for write-back.
    """
    def __init__(self, name: str, data: Any, device: DeviceHandle | None = None):
        self.name = name
        self.snapshot = data                    # comparison value (torch.Tensor)
        self.working = None                     # live execution value
        self.device: DeviceHandle | None = device

    def set_working(self, data=None): ...       # sets working (from snapshot if None)
    def write_to_device(self, program_context): ...  # asserts device, writes working to pool
    def read_from_device(self, program_context): ... # asserts device, reads from pool to snapshot
```

**GoldenMapTensor integration:** Golden `TensorValue.working` stores
`GoldenMapTensor` (from `tools/golden/mapping.py`) natively, so the golden
executor reads it directly without wrapping at call sites. `snapshot` stays as
`torch.Tensor` for PCC comparison.

**`TensorPool`** — dict subclass for tensor storage with optional disk caching:

```python
class TensorPool(dict):
    def __init__(self, caching: bool = False, output_dir: Path | None = None):
        super().__init__()
        self.caching = caching
        self.output_dir = output_dir
        if caching and output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
```

## Porting Notes

### `ops.py` from `runtime/tools/chisel/chisel/core/ops.py`

**`get_op_outputs()` and `get_op_inputs()`:**
- **Port as-is** — pure MLIR operation introspection, no ExecutionType dependency

**`IRModule` changes:**
- **Remove** `execution_type: ExecutionType` constructor parameter and attribute
- **Remove** `self.execution_type` — used only in `__repr__` and passed to AsmState
- **Keep as-is:** `get_function()`, `get_function_inputs()`, `get_function_ops()`,
  `get_asm_state()`, `last_loc_line` property
- The `ignored_ops` parameter stays — useful for skipping `ttnn.deallocate` and
  similar non-compute ops

**`hash_location` and `UNKNOWN_LOCATION`:**
- **Inline from** `runtime/tools/chisel/chisel/utils/location.py` into `ops.py`
- Avoids needing a separate `utils.py` at this stage

### `tensors.py` from `runtime/tools/chisel/chisel/core/tensors.py`

**`TensorValue` → composition refactor:**
- **Replace** with composition design: `DeviceHandle` dataclass + refactored `TensorValue`
- **Remove** `execution_type` parameter — pool identity determines golden vs device
- **Rename** `data` → `snapshot`, `execution_data` → `working`
- **Rename** `set_execution_data()` → `set_working()`
- **Replace** `update_tensor_in_pool()` → `write_to_device()` (asserts `self.device`)
- **Replace** `retrieve_tensor_from_pool()` → `read_from_device()` (delegates to `self.device`)
- **Extract** `tensor_ref` and cached `Tensor` into `DeviceHandle` dataclass
- **Add** `GoldenMapTensor` from `tools/golden/mapping.py` as the type stored in
  `working` for golden tensors
- Late `tensor_ref` assignment pattern becomes: `tv.device = DeviceHandle(tensor_ref)`

**`TensorPool` changes:**
- **Port as-is** — no ExecutionType dependency. The dict subclass with optional
  caching works unchanged.
- Update caching to use `value.snapshot` instead of `value.data`
- In the new design, two `TensorPool` instances exist: `golden_tensor_pool`
  (CPU tensors) and `device_tensor_pool` (hardware tensors), but neither needs
  an ExecutionType tag — the pool identity is determined by which pool the
  context holds.

## Test Plan

### `test_tensors.py`
- `test_tensor_value_creation()` — create with name and data, verify `snapshot`, `working=None`, `device=None`
- `test_device_handle_creation()` — create `DeviceHandle` with mock `tensor_ref`
- `test_tensor_value_with_device()` — create with `device=DeviceHandle(ref)`, verify device field
- `test_set_working_default()` — verify `set_working()` copies snapshot to working
- `test_set_working_explicit()` — verify `set_working(data)` stores given data
- `test_tensor_pool_insert_retrieve()` — insert TensorValue, retrieve by key
- `test_tensor_pool_disk_caching()` — create pool with `caching=True` and tmp dir,
  insert tensor, verify file written to disk (uses `value.snapshot`)
- `test_tensor_pool_is_dict()` — verify TensorPool behaves as dict (keys, values, items)

**Test dependencies:** `torch` only — mock runtime tensors where needed.

### `test_ops.py`
- `test_ir_module_creation()` — parse a small TTNN MLIR module string, create IRModule
- `test_get_function()` — verify `get_function()` returns the expected function op
- `test_get_function_ops()` — verify operations are listed in correct order
- `test_last_loc_line()` — verify location-to-index mapping
- `test_get_op_inputs_outputs()` — verify tensor-like operand/result extraction
- `test_ignored_ops()` — verify ops in `ignored_ops` list are filtered
- `test_hash_location()` — mock MLIR Location objects, verify consistent hashing
- `test_unknown_location_constant()` — verify `UNKNOWN_LOCATION == (-1, -1)`

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
