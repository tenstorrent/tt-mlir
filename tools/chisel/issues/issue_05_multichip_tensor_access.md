# Multi-Device Tensor Access via retrieve_tensor_from_pool

## Description

Update the existing `retrieve_tensor_from_pool` runtime API to return a map of
logical device ID → per-device host tensor shard, instead of a single merged
host tensor. Currently, the C++ implementation
(`runtime/lib/ttnn/runtime.cpp:2032`) already calls `toHost()` which returns
`std::vector<Tensor>`, but then fatals if `hostTensors.size() != 1`. The fix is
to return the full set of shards keyed by logical device ID, following the same
pattern already used by `getOpOutputTensor` (`runtime/lib/ttnn/runtime.cpp:958`)
which returns `std::unordered_map<std::uint32_t, Tensor>`. This enables Chisel
debug hooks to inspect or override individual device shards (e.g., tracing
per-device shapes, replacing output on specific devices).

## Current Behavior

- `retrieveTensorFromPool` returns `std::optional<Tensor>` (single tensor)
- Calls `toHost()` → gets `std::vector<Tensor>`
- `LOG_FATAL` if vector has more than one element (multi-device not supported)
- Python binding returns `Optional[tt.runtime.Tensor]`

## Proposed Change

Change the return type to `std::unordered_map<std::uint32_t, Tensor>` where
keys are logical device IDs (indices into the `toHost()` vector, matching the
row-major ordering convention documented in `getOpOutputTensor`).

### C++ signature change

```cpp
// Before
std::optional<Tensor>
retrieveTensorFromPool(CallbackContext programContextHandle,
                       TensorRef tensorRef, bool untilize);

// After
std::unordered_map<std::uint32_t, Tensor>
retrieveTensorFromPool(CallbackContext programContextHandle,
                       TensorRef tensorRef, bool untilize);
```

### Implementation

Replace the `LOG_FATAL` guard with a loop that maps each shard to its logical
device index, mirroring the existing `getOpOutputTensor` implementation:

```cpp
std::unordered_map<std::uint32_t, Tensor> result;
std::vector<tt::runtime::Tensor> hostTensors =
    ::tt::runtime::ttnn::toHost(outTensor, untilize);
for (size_t i = 0; i < hostTensors.size(); ++i) {
  result[i] = hostTensors[i];
}
return result;
```

### Python binding change

Update `runtime/python/runtime/runtime.cpp` binding to return
`Dict[int, tt.runtime.Tensor]` instead of `Optional[tt.runtime.Tensor]`.

### Existing precedent

`getOpOutputTensor` (`runtime/lib/ttnn/runtime.cpp:958`) already returns
`std::unordered_map<std::uint32_t, Tensor>` using the same logical-device-ID
keying convention with the same `toHost()` decomposition and index-based loop.
