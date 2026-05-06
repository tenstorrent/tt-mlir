# Tensor Liveness in the Runtime

This document explains how to determine whether a tensor is still "live"
(allocated and present in the runtime pool) given its compile-time global ID.
This is relevant to Chisel's `global_tensor_pool`, which must detect when
device tensors have been deallocated between programs.

## Two Different Global IDs

The runtime has two distinct ID spaces — do not confuse them:

| ID | Type | Source | Purpose |
|----|------|--------|---------|
| `TensorRef.global_id` | `uint32` | Compile-time (flatbuffer) | Key in `ProgramTensorPool.liveTensors` |
| `Tensor::globalId` | `uint64` | Runtime (atomic counter) | Process-wide unique ID, used by distributed command system |

The **flatbuffer `TensorRef.global_id`** is the one used for pool lookups.
The **runtime `Tensor::globalId`** has no reverse-lookup registry.

## ProgramTensorPool Liveness

`ProgramTensorPool` (`runtime/include/tt/runtime/detail/ttnn/types/types.h`)
maintains two maps, both keyed by `TensorRef.global_id` (uint32):

```cpp
TensorMap intermedTensors;   // owns intermediate tensors
TensorPtrMap liveTensors;    // pointers to all live tensors
```

### Checking if a tensor is live

```cpp
bool contains(const TensorRef *tensorRef) const {
    return liveTensors.contains(tensorRef->global_id());
}
```

Or directly by raw ID:

```cpp
const Tensor &getRuntimeTensor(uint32_t globalId) const;  // private, but available internally
```

### When a tensor becomes dead

`DeallocateOp` (`runtime/lib/ttnn/operations/deletion/deallocate.cpp`) does:

1. Checks `TTNNTensorWrapper::shouldRetain()` — if true, skips deallocation.
2. Calls `::ttnn::deallocate(tensor, force)` to free device memory.
3. Calls `tensorPool.erase(tensorRef)` — **removes from `liveTensors`**.

After `erase`, `contains()` returns false for that `global_id`.

### Scope

`ProgramTensorPool` is **per-program-execution** — it is created when a
program starts and destroyed when it ends. There is no global registry of all
live tensors across programs.

## Implication for Chisel's global_tensor_pool

Chisel's `global_tensor_pool` is keyed by `Tensor::globalId` (uint64) and
persists across programs. The problem: a tensor stored in the global pool may
have been deallocated by the runtime between program executions.

To detect stale entries, Chisel should check whether the tensor's
`TensorRef.global_id` is still present in the current `ProgramTensorPool`
during `preProgram`. If a program's input tensor IDs (from
`get_program_input_ids()`) do not include a given `global_id`, that tensor
was not carried forward and should not be assumed live on device.

In practice:
- **preProgram**: Copy only tensors whose `TensorRef.global_id` appears in
  the new program's input ID list.
- **postProgram**: Copy program output tensors to `global_tensor_pool` for
  cross-program reuse.
- Tensors that were deallocated mid-program (via `DeallocateOp`) will not
  appear in output IDs and should not be copied back.
