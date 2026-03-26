# Multi-Chip Support in Builder — Findings for Chisel

## Context

Research into how the builder handles multi-chip/multi-device execution, to inform
Chisel's multi-chip support design.

## How the Builder Handles Multi-Chip

### 1. Per-Device Golden Tensors via `GoldenMapTensor`

**File:** `tools/golden/mapping.py:26-100`

- Each golden tensor is a `GoldenMapTensor` wrapping `shard_map: Dict[int, torch.Tensor]`
  keyed by **logical device ID**
- Single-device: key `0` only. Multi-device: one entry per chip
- `__torch_function__` protocol applies torch ops independently to each shard
- `mesh_shape: Tuple[int, int]` tracks device topology per tensor

### 2. Multi-Device Tensor Creation

**File:** `tools/builder/base/builder_runtime.py:112-134`

```python
if len(tensor_shards.keys()) > 1:
    return tt_runtime.runtime.create_multi_device_borrowed_host_tensor(
        [t.data_ptr() for t in tensor_shards.values()],
        list(first_tensor.shape),
        list(first_tensor.stride()),
        first_tensor.element_size(),
        torch_dtype_to_runtime_dtype(first_tensor.dtype),
        {},          # strategy: not used
        mesh_shape,  # from device.get_mesh_shape()
    )
```

### 3. Per-Device Golden Comparison in Callbacks

**File:** `tools/builder/base/builder_runtime.py:452-543`

```python
def golden(callback_runtime_config, binary, program_context, op_context):
    op_output_tensor_map = tt_runtime.runtime.get_op_output_tensor(op_context, program_context)
    # Returns Dict[device_id, Tensor] — one tensor per chip

    for device_id, golden_tensor_torch in op_golden_tensor_map.items():
        if device_id not in op_output_tensor_map.keys():
            continue
        op_output_tensor = op_output_tensor_map[device_id]
        # Per-device comparison...
        device_results[device_id] = results

    callback_runtime_config.golden_report[loc] = device_results
```

### 4. Runtime Per-Device Output Extraction

**File:** `runtime/lib/ttnn/runtime.cpp:958-999`

`getOpOutputTensor()` returns `unordered_map<uint32_t, Tensor>`:
- Looks up the TTNN tensor in the `ProgramTensorPool` via `TensorRef`
- Calls `toHost()` which splits multi-device tensor into per-device shards
- Indexes by logical device ID (row-major order, NOT physical device IDs)

### 5. Device Mesh Setup

```python
mesh_options = tt_runtime.runtime.MeshDeviceOptions()
mesh_options.mesh_shape = (1, 2)  # e.g., 2 chips
device = tt_runtime.runtime.open_mesh_device(mesh_options)
device.get_mesh_shape()  # returns (rows, cols)
```

### 6. Program-Level Mesh Shape

**File:** `runtime/python/binary/binary.cpp:49`

```python
binary.get_program_mesh_shape(program_index)  # returns (y, x) pair
```

## Tensor Shape Information in the Flatbuffer

### TensorDesc Structure

**File:** `include/ttmlir/Target/TTNN/types.fbs:94-100`

```
table TensorDesc {
  shape: [int];            // Global (logical) shape — full tensor before distribution
  mesh_shape: [int32];     // Device mesh dimensions, e.g. [1, 2]
  layout: LayoutDesc;
  shard_status: ShardStatus;  // Presharded or Unsharded
  local_shape: [int];      // Per-device shape after sharding
}

table TensorRef {
  global_id: uint32;
  desc: TensorDesc;
}
```

### Available for ALL Tensors (Inputs, Outputs, AND Intermediates)

Every `TensorRef` in every operation carries the full `TensorDesc`. From C++:

```cpp
const auto *tensorRefPtr = &tensorRef.as<tt::target::ttnn::TensorRef>(DeviceRuntime::TTNN);
tensorRefPtr->desc()->shape()        // global shape
tensorRefPtr->desc()->local_shape()  // per-device shard shape
tensorRefPtr->desc()->mesh_shape()   // mesh topology [rows, cols]
tensorRefPtr->desc()->shard_status() // Presharded or Unsharded
```

### NOT Exposed to Python

**File:** `runtime/python/runtime/runtime.cpp:247`

```cpp
nb::class_<tt::runtime::TensorRef>(m, "TensorRef");  // opaque — no properties
```

`TensorRef` has zero `.def()` calls in Python bindings. To access `desc()` from
Python, new bindings would need to be added (e.g., `get_desc_shape()`,
`get_desc_local_shape()`, `get_desc_mesh_shape()`).

**Workaround:** `binary.get_program_ops_as_json(program_index)` serializes all
operations including TensorRef descs as JSON — available today but requires
parsing the full program and indexing by op location.

## Key Caveats

1. **`retrieve_tensor_from_pool()` does NOT support multi-device** —
   `runtime/lib/ttnn/runtime.cpp:1990-1992` fatally errors if `hostTensors.size() != 1`.
   Only `get_op_output_tensor()` handles multi-device (returns per-device map).

2. **`TensorRef` Python bindings are opaque** — shape/local_shape/mesh_shape
   from the flatbuffer desc are not accessible from Python without new bindings.

3. **Logical vs physical device IDs** — `get_op_output_tensor()` returns tensors
   indexed by logical device ID (row-major), not physical device IDs.

## How Serialization Populates Shape Info

**File:** `lib/Target/TTNN/TTNNToFlatbuffer.cpp:143-202`

- `shape` comes from the MLIR `RankedTensorType` shape (global)
- `mesh_shape` is `{1,1}` for single-device; `deviceAttr.getMeshShape()` for multi-device
  (set when `layoutAttr.getTensorMesh()` is present)
- `local_shape` comes from the `localShapeType` parameter (per-device shard shape)
- `shard_status` maps from `ttcore::ShardStatus::Presharded` / `Unsharded`
