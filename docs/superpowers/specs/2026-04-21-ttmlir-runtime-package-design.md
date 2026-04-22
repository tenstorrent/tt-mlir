# Design: `ttmlir_runtime` Standalone Python Package

**Date:** 2026-04-21
**Author:** Nikola Drakulic

## Problem

`_ttmlir_runtime.so` is currently owned by the `ttrt` pip package — it is
physically placed inside `ttrt/runtime/` during build and loaded by Python as
`ttrt.runtime._ttmlir_runtime`.

`builder` and `chisel` both do `import _ttmlir_runtime as tt_runtime` directly,
which finds the same `.so` elsewhere on `sys.path` and loads it a second time
under the module name `_ttmlir_runtime`. Python treats these as two distinct
modules even though they share one `.so`. Nanobind's per-interpreter type
registry is populated twice with conflicting entries, causing segfaults whenever
objects from one module instance are passed to code holding a handle from the
other.

## Goal

Create a standalone `ttmlir_runtime` pip package that owns the `.so` and all
runtime dylibs. `ttrt` (CLI), `builder`, and `chisel` all import from
`ttmlir_runtime`. The `.so` is loaded exactly once, under the single module path
`ttmlir_runtime._ttmlir_runtime`.

## Non-goals

- Changing the C++ nanobind source (`__init__.cpp`, `runtime/runtime.cpp`,
  `binary/binary.cpp`, etc.) — these are untouched.
- Merging or deduplicating the Python utility layer (dtype converters, PCC
  functions, `CallbackRuntimeConfig`) — that is a separate cleanup.
- Moving `ttrt/binary/stats.py` — it is ttrt-CLI-specific and stays in `ttrt`.

## Package Layout

```
runtime/python/
├── __init__.cpp              # existing NB_MODULE — unchanged
├── binary/                   # existing C++ source — unchanged
├── runtime/                  # existing C++ source — unchanged
├── ttmlir_runtime/           # NEW Python package
│   ├── __init__.py
│   ├── runtime/
│   │   └── __init__.py       # re-exports from _ttmlir_runtime.runtime
│   ├── binary/
│   │   └── __init__.py       # re-exports from _ttmlir_runtime.binary + json helpers
│   └── utils/
│       └── __init__.py       # conditional re-export from _ttmlir_runtime.utils
└── setup.py                  # NEW — owns .so copy and dylib packaging
```

The `.so` file and all dylibs are placed inside `ttmlir_runtime/runtime/`
(mirroring the current `ttrt/runtime/` layout), so Python loads the extension
as `ttmlir_runtime.runtime._ttmlir_runtime`. The re-export wrappers in
`ttmlir_runtime/runtime/__init__.py` and `ttmlir_runtime/binary/__init__.py`
then pull symbols from that single load path. No other import path for this
module exists after the migration.

## Consumer Changes

### `ttmlir_runtime/runtime/__init__.py`

Re-exports everything currently in `ttrt/runtime/__init__.py`:

```python
from ttmlir_runtime._ttmlir_runtime.runtime import (
    Device, Event, Tensor, TensorRef, TensorDesc,
    MemoryBufferType, DataType, DeviceRuntime, HostRuntime,
    DebugHooks, DebugEnv, PerfEnv, DebugStats,
    MeshDeviceOptions, MultiProcessArgs, DistributedOptions, DistributedMode,
    set_mlir_home, set_metal_home, get_current_system_desc,
    open_mesh_device, close_mesh_device, submit, wait,
    to_host, to_layout, get_layout, memcpy, deallocate_tensor,
    create_borrowed_host_tensor, create_owned_host_tensor,
    create_empty_tensor, create_multi_device_borrowed_host_tensor,
    get_op_output_tensors, get_op_output_refs, get_op_input_refs,
    retrieve_tensor_from_pool, update_tensor_in_pool,
    get_op_loc_info, get_op_debug_str, unregister_hooks,
    get_num_available_devices, get_num_shards, get_device_tensors,
    WorkaroundEnv, FabricConfig, DispatchCoreType,
    # ... (full list mirrors current ttrt/runtime/__init__.py)
)

try:
    from ttmlir_runtime._ttmlir_runtime.runtime import test
except ImportError:
    pass
```

### `ttmlir_runtime/binary/__init__.py`

Re-exports the C++ binary API plus the JSON helper utilities currently in
`ttrt/binary/__init__.py`:

```python
from ttmlir_runtime._ttmlir_runtime.binary import (
    load_from_path, load_binary_from_path, load_binary_from_capsule,
    load_system_desc_from_path, Flatbuffer, GoldenTensor,
)

import json, re

def json_string_as_dict(json_string): ...
def fbb_as_dict(bin): ...
def system_desc_as_dict(bin): ...
def program_ops_as_dict(bin, index): ...
def program_inputs_as_dict(bin, index): ...
def program_outputs_as_dict(bin, index): ...
def mlir_as_dict(bin): ...
```

### `ttmlir_runtime/utils/__init__.py`

Conditional re-export (mirrors existing behaviour):

```python
try:
    from ttmlir_runtime._ttmlir_runtime.utils import *
except ImportError:
    pass
```

### `ttrt`

- `ttrt/runtime/__init__.py` becomes a shim:
  ```python
  from ttmlir_runtime.runtime import *
  ```
- `ttrt/binary/__init__.py` becomes a shim:
  ```python
  from ttmlir_runtime.binary import *
  from . import stats   # stats stays in ttrt
  ```
- `ttrt/setup.py`: remove all `.so` / dylib copy logic; add
  `ttmlir_runtime` to `install_requires`.

### `builder`

`tools/builder/base/builder_runtime.py` line 20:

```python
# before
import _ttmlir_runtime as tt_runtime

# after
import ttmlir_runtime as tt_runtime
```

No other changes — `tt_runtime.runtime.DataType`, `tt_runtime.binary.load_binary_from_capsule`, etc. continue to work because `ttmlir_runtime` re-exports those submodules identically.

### `chisel`

`tools/chisel/chisel/bind.py` line 15:

```python
# before
import _ttmlir_runtime as tt_runtime

# after
import ttmlir_runtime as tt_runtime
```

## Build System

### `runtime/python/setup.py` (new)

Takes over the `.so` copy and dylib packaging logic from `ttrt/setup.py`:

- Copies `_ttmlir_runtime.cpython-*.so` → `ttmlir_runtime/runtime/`
- Copies `libTTMLIRRuntime.so` → `ttmlir_runtime/runtime/`
- Copies `_ttnncpp.so`, `libtt_metal.so`, `libtt-umd.so`, `libtt_stl.so`,
  `libtracy.so.0.10.0` → `ttmlir_runtime/runtime/`
- Copies `tt_metal/`, `runtime/` (metal runtime data), `ttnn/` kernel folders
  → `ttmlir_runtime/runtime/`  (same flat layout as current `ttrt/runtime/`)
- `package_data`: all dylibs and data folders under `ttmlir_runtime/runtime/`

All import paths in the re-export wrappers use
`ttmlir_runtime.runtime._ttmlir_runtime.*` (not `ttmlir_runtime._ttmlir_runtime.*`).

### `runtime/python/CMakeLists.txt`

Add a new `ttmlir_runtime` Python packaging target (mirrors the existing `ttrt`
packaging target). The existing `nanobind_add_module(_ttmlir_runtime ...)` build
target is unchanged; only the destination directory for packaging changes.

### Install order

```
build _ttmlir_runtime.so
  → package into ttmlir_runtime wheel  (runtime/python/setup.py)
  → pip install ttmlir_runtime
  → pip install ttrt  (declares ttmlir_runtime in install_requires)
```

## Migration Checklist

- [ ] Create `runtime/python/ttmlir_runtime/` package with `__init__.py`,
      `runtime/__init__.py`, `binary/__init__.py`, `utils/__init__.py`
- [ ] Create `runtime/python/setup.py` with `.so` + dylib copy logic
- [ ] Add CMake packaging target for `ttmlir_runtime`
- [ ] Update `ttrt/runtime/__init__.py` → shim
- [ ] Update `ttrt/binary/__init__.py` → shim (keep `stats` import)
- [ ] Update `ttrt/setup.py` → remove copy logic, add `ttmlir_runtime` dep
- [ ] Update `builder/base/builder_runtime.py` → `import ttmlir_runtime`
- [ ] Update `chisel/bind.py` → `import ttmlir_runtime`
- [ ] Verify no remaining `import _ttmlir_runtime` outside of `ttmlir_runtime/`
- [ ] Run existing tests to confirm no regressions
