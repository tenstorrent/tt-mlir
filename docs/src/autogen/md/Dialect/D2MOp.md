# D2M Dialect Ops

Auto-generated reference of operations in the `D2M` dialect (126 ops).

## `d2m.::llvm::cast<::mlir::tt::ttcore::MemorySpaceAttr>(::llvm::cast<::mlir::MemRefType>($_self).getMemorySpace()).getValue() == ::mlir::tt::ttcore::MemorySpace::DeviceL1`

## `d2m.acquire_dst`

Acquire Destination Register op.

This op returns a memref with memory space dest that models the destination register
resource on tensorrent hardware. Example IR:

```mlir
%dst = d2m.acquire_dst() : memref<2x4x!tt.tile<32x32, f32>, #tt.memory_space<dst>>
```

## `d2m.arange_block`

Generate arange values in a block/memref

The `arange_block` operation fills a block/memref with arange values:
output[i] = start + step * i, where i is the linear index.

This op operates at the block level (memref of tiles).

The index_tile_tensor is a scratch tensor (1x1 tile) handled by
the FillArangeTileOp.

The colMajor option specifies whether values should be filled in a column major order instead (default is row major).

For each tile at position (tile_row, tile_col), the computation is:
1. tile_offset = (tile_row * num_tile_cols * 32 + tile_col) * 32
2. global_idx = tile_add(tile_offset, local_idx)
3. result = tile_add(tile_mul(global_idx, step), start)

If colMajor is true, then the computation is:
1. tile_offset = (tile_col * num_tile_rows * 32 + tile_row) * 32
2. global_idx = tile_add(tile_offset, local_idx)
3. result = tile_add(tile_mul(global_idx, step), start)

Example:
```mlir
%result = d2m.arange_block %output, %index_tile
                  {num_elements = 64, start = 0, step = 1}
    : (tensor<1x2x!ttcore.tile<32x32, f32>>,
        tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x2x!ttcore.tile<32x32, f32>>
```

## `d2m.block_index`

Block Index op.

Return the index for the given block dimension. This op represents
a symbolic index that requires grid/block calculation during
loop generation. It is created when expanding indexing map expressions
and requires combining grid indices with block factors and loop indices
to compute the final value.

## `d2m.block_offset`

Block Offset op.

Represents block_factor * core_index as a constant-like operation. Useful
for representing blocking index math in an affine dialect compatible way.

## `d2m.composite_view`

Piecewise view over multiple input tensors

Represents a piecewise-affine view that aggregates multiple input tensors
along a certain dimension.
The optional `logicalSizes` attribute records the logical sizes of the
inputs along `dim`, which is required for row-major inputs.

## `d2m.core_index`

Core Index op.

Return the index of this core's coordinate inside the generic op's grid dimension.

## `d2m.create_global_semaphore`

Global semaphore allocation operation (D2M).

Create an uninitialized global semaphore with the specified core range.

## `d2m.create_local_semaphore`

Create local semaphore op.

This operation is a representational op. It creates a local semaphore in the core range within the generic op in which it is consumed.

## `d2m.device_synchronize`

Device synchronize op.

The different devices in a mesh are not running ops in a synchronized manner
(devices can be running different ops in the program) so in ops where we are transferring data between devices,
we need to ensure that receiving devices have started the op before senders send data to them.

This op allows us to do this in two steps:
1) receiver devices inform senders that they have started the op by incrementing the sync_semaphore on the sender device
2) senders wait for all receivers that they have to send to using a semaphore wait

To ensure correct behaviour, we have to correctly set senderStartDevice, senderDeviceMcastShape, and
numReceivers based on which devices are sending to which receivers. Note that the senders and receivers
to be specified are in terms of the own device's perspective.

Example 1: All gather on a 1x8 mesh along cluster axis 1:
  We have to set senderStartDevice to [0, 0], senderDeviceMcastShape to [1, 8], and numReceivers to 7 for ALL devices.

Example 2: broadcast CCL from device x to all other devices in 1x8 mesh:
  - on device x, senderStartDevice/senderDeviceMcastShape will be empty since it is not receiving any data from other devices, but
  numReceivers will be 7 since it is sending to all other devices
  - on other devices, senderStartDevice will be device x, senderDeviceMcastShape will be empty, and numReceivers will be 0 (not sending to anyone)

Pre-condition: sync_semaphore is initialized to 0.
Post-condition: Receiver devices are guaranteed to have started their op
  and so we can send data to them.

## `d2m.dma_read`

Lowered D2M DMA Read Op.

This op performs a DMA read operation from src memref to dst memref.
The dst memref must be local. The src memref can either be remote or local.

It supports two forms distinguished by numElems:
Fully indexed form (numElems > 0): src and dst indices are fully
resolved (all view and memspace-specific affine maps applied). Corresponds
1:1 to a single concrete DMA read operation.

Remote-to-local fully indexed form:
```mlir
%tx = d2m.dma_read %src[%gridy, %gridx, %offset], %dst[%offset], <8>
  : (memref<2x2x2x4x!ttcore.tile<32x32, f32>, #dram>,
     memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
```

Local-to-local fully indexed form:
```mlir
%tx = d2m.dma_read %src[%offset], %dst[%offset], <8>
  : (memref<2x4x!ttcore.tile<32x32, f32>, #l1>,
     memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
```

Shard-level form (numElems == 0): indices are grid-level only on the
remote src operand, no indices on the local dst. Represents "read entire
shard at grid position". The indexing expansion is deferred to
D2MLowerDMAToFullyIndexedForm.

```mlir
%tx = d2m.dma_read %src[%grid_i, %grid_j], %dst, <0>
  : (memref<2x2x2x4x!ttcore.tile<32x32, f32>, #dram>,
     memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
```

This operation only supports unicast (multicast are write-only).

Constraints:
- dst MUST be a LOCAL memref!
- src and dst must have the same element type.

## `d2m.dma_wait`

D2M DMA wait Op

Waits for the producer DMA memory transaction to complete.

## `d2m.dma_write`

Lowered D2M DMA Write Op

This op performs a DMA write operation from src memref to dst memref. This operation supports multicast; providing an empty multicast shape
argument _implies a unicast write_.

It supports two forms distinguished by numElems:
Fully indexed form (numElems > 0): src and dst indices are fully
resolved (all view and memspace-specific affine maps applied). Corresponds
1:1 to a single concrete DMA write operation.

```mlir
%tx = d2m.dma_write %src[%offset], %dst[%gridy, %gridx, %offset], <%size>
  : (memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x2x2x4x!ttcore.tile<32x32, f32>, #dram>) -> !d2m.mem_tx
```

Shard-level form (numElems == 0): indices are grid-level only.
Represents "write entire shard at grid position". The indexing expansion
is deferred to D2MLowerDMAToFullyIndexedForm.

```mlir
%tx = d2m.dma_write %src, %dst[%grid_i, %grid_j], <0>
  : (memref<2x4x!ttcore.tile<32x32, f32>, #l1>,
     memref<2x2x2x4x!ttcore.tile<32x32, f32>, #dram>) -> !d2m.mem_tx
```

Constraints:
- src MUST be a LOCAL memref!
- src and dst must have the same element type.

## `d2m.dst_reinterpret_cast`

D2M Destination Register Reinterpret Cast Op

The `dst_reinterpret_cast` operation reinterprets a tile value as a different
tile type for the purpose of storing/loading from the destination register.

This models the hardware behavior that destination register memory is untyped
and can be accessed with different type interpretations. It is inserted during
the InsertDstRegisterAccess pass to reconcile type mismatches between compute
operations and destination register allocations, and is removed during lowering
to TTKernel as a no-op.

Example:
```mlir
%0 = affine.load %dst : memref<1x1x4x!ttcore.tile<32x32, f32>, #dst>
%1 = d2m.tile_typecast %0 : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, bf16>
%2 = d2m.dst_reinterpret_cast %1 : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, f32>
affine.store %2, %dst : memref<1x1x4x!ttcore.tile<32x32, f32>, #dst>
```

## `d2m.embedding`

D2M embedding row-gather op

Data-movement embedding lookup. The op reads integer indices and a row-major
embedding table, then writes the selected table rows to the output buffer.

The tensor form is emitted by TTIRToD2M with an output tensor operand and
a tensor result. During bufferization, this op allocates scratch circular
buffers and lowers to `d2m.indexed_row_copy`, which owns the backend row
gather implementation.

## `d2m.fill_arange_tile`

Write full linear index pattern to a tile CB

The `fill_arange_tile` operation writes a linear index pattern
to a tile CB, where element[i,j] = i * 32 + j in the tile's native data
format (f32, bf16, or i32).

This is used for arange operations. Each element gets its linear index value
(0-1023) within the tile, which can be used to compute global indices.

The resulting tile looks like:
[[  0,   1,   2, ...,  31],
  [ 32,  33,  34, ...,  63],
  [ 64,  65,  66, ...,  95],
  ...
  [992, 993, 994, ..., 1023]]

## `d2m.get_arg`

Get a generic argument for the given operand index.

Obtain a handle for the specified operand index. The result type determines
how the argument is interpreted: a memref type indicates a buffer address,
a local/global semaphore type indicates a semaphore.

Used to reference additional arguments passed to the parent generic op
from within a thread region or a standalone kernel function.

```mlir
%buf   = d2m.get_arg(0) : memref<...>
%lsem  = d2m.get_arg(3) : !d2m.local_semaphore
%gsem  = d2m.get_arg(4) : !d2m.global_semaphore
```

## `d2m.get_block_factor`

Get block factor op.

Retrieve the block factor from the block factor array attribute in
the parent generic op at the provided dimension index.

## `d2m.get_cb`

Get a circular buffer for the given cb operand index.

Obtain a CB handle for the specified cb operand in the generic.

The cb_operand_idx corresponds to the parent generic op's operand index.

```mlir
%cb0 = d2m.get_cb(0) : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>
%buf = d2m.wait %cb0 : !d2m.cb<memref<...>> -> memref<...>
```

## `d2m.indexed_row_copy`

D2M indexed row-copy op

Data-movement primitive for copying rows from a source table to an output
buffer using a tensor of 32-bit row indices. This is the lower-level
operation used by embedding, and is shaped so other indexed
datamovement ops can share the same NoC lowering path.

The op is memref-only. Tensor-level semantic ops should bufferize or
decompose to this op after choosing scratch circular buffers.

## `d2m.iter_index`

Iter Index op.

Return the index of the current element in the iteration for the given generic op dimension.

## `d2m.local_copy`

Local DMA copy with affine map indexing

Copies data between two local L1 buffers using affine maps to describe
the source/destination indexing relationship.  The iteration domain is
the destination shape; `LowerDMAToFullyIndexedForm` generates the
actual loops and point-access `d2m.dma_read` ops from the maps.

Both operands must be local (non-DRAM).

The operation has three surface forms:

Tensor form (pre-bufferization):
```mlir
%result = d2m.local_copy %src, %dst indexing_maps = [#src_map, #dst_map]
  : tensor<...>, tensor<...> -> tensor<...>
```

Memref form (post-bufferization, no result):
```mlir
d2m.local_copy %src, %dst indexing_maps = [#src_map, #dst_map]
  : memref<...>, memref<...>
```

Explicit CB form (after SplitUnifiedThread):
```mlir
d2m.local_copy from %srcCb into %dstCb indexing_maps = [#src_map, #dst_map]
  : from !d2m.cb<memref<...>> into !d2m.cb<memref<...>>
```

Memref form with MemTx result (after LowerLoadStoreOpsToDMA):
```mlir
%tx = d2m.local_copy %src, %dst indexing_maps = [#src_map, #dst_map]
  : memref<...>, memref<...> -> !d2m.mem_tx
d2m.dma_wait %tx
```

## `d2m.mesh_position`

Mesh position op.

Return the mesh position for the current device at the given dimension.

## `d2m.null_tx`

Create a null transaction.

Utility op to create a null transaction.  This is required for creating a sentinel
starting transaction for a DMA nested inside of a loop nest.

## `d2m.operand_alias`

Alias a shard of an input buffer without allocation.

The `alias_buffer` operation declares that a local buffer within a generic
region directly aliases (shares storage with) a shard of an input operand
rather than requiring a separate allocation. It takes a full sharded memref
(with grid dimensions) and returns a memref containing only the shard
portion (grid dimensions stripped).

This op is introduced by the allocator pass for operands that do not
require streaming. It encodes in the IR that no separate buffer
allocation is needed; the compute kernel operates directly on the
input data in place. Downstream passes convert this op into circular
buffer operations (reserve/push/wait/pop).

```mlir
d2m.generic ins(%input : memref<1x1x32x32xf32, ...>) ... {
  %shard = d2m.alias_buffer %input :
      memref<1x1x32x32xf32, ...> -> memref<32x32xf32, ...>
}
```

## `d2m.pop`

Pop from circular buffer (signal consumer done).

Pop operation, releases the memref acquired via d2m.wait.
Signals to producer threads that space is available.
Must be preceded by d2m.wait in the same block.

This operation explicitly releases the circular buffer slot acquired by
d2m.wait, making space available for producer threads. If d2m.pop is
present, the automatic release at block end is skipped for the
corresponding d2m.wait operation.

Example:
```mlir
%memref = d2m.wait %cb : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
// ... read data from %memref ...
d2m.pop %cb : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
```

## `d2m.print`

Print values from a kernel thread.

std::format-style format string with `{}` placeholders, lowered to
ttkernel.dprint. Zero or more values may be interpolated.

Example:
```
d2m.print("semaphore addr = {}\n", %sem) : (!d2m.local_semaphore)
d2m.print("x={} y={}\n", %x, %y) : (ui32, ui32)
d2m.print("done\n") : ()
```

Notes:
  - Only the trivial `{}` format specifier is currently supported.
  - Must double-escape newlines and other special characters.

## `d2m.push`

Push to circular buffer (signal producer done).

Push operation, releases the memref acquired via d2m.reserve.
Signals to consumer threads that data is ready.
Must be preceded by d2m.reserve in the same block.

This operation explicitly releases the circular buffer slot acquired by
d2m.reserve, making the data available to consumer threads. If d2m.push
is present, the automatic release at block end is skipped for the
corresponding d2m.reserve operation.

Example:
```mlir
%memref = d2m.reserve %cb : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
// ... write data to %memref ...
d2m.push %cb : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
```

## `d2m.remote_load`

D2M Remote/Aliased Load Op

Loads an _entire shard_ from remote or local GenericOp operand into a
local L1 buffer. The memref/tensor argument _must_ be an operand of the
GenericOp.

RemoteLoadOp indices correspond to the _grid dimensions only_ (first
N/2 dimensions of the device shape). The RemoteLoadOp always loads an
entire shard from the corresponding operand.

This operation is well-defined for both remote and local tensors. Backend
DMA passes opportunistically lower this operation to aliased CB operations
if possible, but the semantics for remote and local tensors are identical.

The operation requires an explicit destination buffer parameter. The
operation has multiple modes; the explicit CB form and unicast/multicast
modes are both orthogonal and can be mixed and matched.

The basic form takes a local buffer as input and loads data into it.
In tensor form (pre-bufferization), the op produces a result that aliases
the local buffer:

```mlir
%result = d2m.remote_load %buffer %generic_operand[%i, %j]
  : tensor<SY x SX x f32, ...>, tensor<GY x GX x SY x SX x f32, ...>
  -> tensor<SY x SX x f32, ...>
```

In memref form (post-bufferization), the op has no result; data is loaded
in-place into the local buffer:

```mlir
d2m.remote_load %buffer %generic_operand[%i, %j]
  : memref<SY x SX x f32, ...>, memref<GY x GX x SY x SX x f32, ...>
```

The explicit CB form takes a CB (from `d2m.get_cb`) as additional input;
the load is produced into the CB. The CB provided must be the CB associated
with the operand. In explicit CB form, no local buffer is required.

```mlir
d2m.remote_load %generic_operand[%i, %j] into %cb
  : memref<GY x GX x SY x SX x f32, ...> into !d2m.cb<memref<SY x SX x f32, ...>>
```

In implicit tensor form (pre-bufferization), the result matches the shard
tensor type and aliases the input buffer. In implicit memref form
(post-bufferization), there is no result.

## High-Level Multicast Mode

When a high-level multicast dimension list (mcast) is provided, this
operation performs a cooperative multicast along the parallel dimension
specified. All locations in the compute grid that share the same mcast dim value
will form a multicast group *if all non-mcast dims are reduction dims*.

The sender vs receiver roles in the multicast group are determined by convention.

NOTE: If implementing the high-level multicast is not feasible, the
operation will be lowered to an equivalent unicast remote load instead.

Example with high-level multicast (implicit memref form):

```mlir
// Cooperative multicast along dimension 0
d2m.remote_load %buffer %memref[%i, %j] mcast[%c0]
  : memref<4x6xf32, #l1_>, memref<2x4x4x6xf32, #ttcore.shard<...>, #dram>
```

## Low-Level Multicast Mode

WARNING: This form is currently supported mostly for internal use within
middle-end pipeline. Picking a concrete multicast shape implicitly
constrains that the layout of the operand, but this isn't handled properly
in the frontend grid selection.

When low-level multicast parameters (mcore/mshape) are provided, this
operation performs a gather-multicast pattern:
  - By convention, core 0 along each multicast dimension is the sender
  - The sender reads the remote data into local memory
  - The sender then multicasts to all other cores in the mcast group
  starting at `mcastStartIndex` with shape `mcastShape`
  - Receivers wait for the sender to complete

Example with low-level multicast (implicit memref form):

```mlir
// Core 0 gathers and multicasts to a 1x4 region starting at mcore[0, 1]
d2m.remote_load %buffer %memref[%i, %j] mcore[%c0, %c1] mshape[%c1, %c4]
  : memref<4x6xf32, #l1_>, memref<2x4x4x6xf32, #ttcore.shard<...>, #dram>
```

Constraints:
- The number of indices must equal N/2, where N is the memref/tensor rank.
- The operation loads the entire shard into the provided local L1 buffer
- If in implicit form (no CB):
  - The localBuffer type must match the shard shape of the memref operand
  - The localBuffer is required
- If in explicit CB form:
  - The operand provided must be a memref
  - The localBuffer must NOT be present
  - The CB underlying type must match the shard shape
- The two multicast forms are mutually exclusive: either use low-level
(mcore/mshape) or high-level (mcast dims), not both

## `d2m.remote_store`

D2M Remote/Aliased Store Op

Stores an _entire shard_ from a local buffer to remote or local GenericOp
operand. The memref/tensor argument _must_ be an operand of the GenericOp.

RemoteStoreOp indices correspond to the _grid dimensions only_ (first N/2
dimensions of the device shape). The RemoteStoreOp always stores an entire
shard to the corresponding operand.

This operation is well-defined for both remote and local tensors. Backend
DMA passes opportunistically lower this operation to aliased CB operations
if possible, but the semantics for remote and local tensors are identical.

The operation has two forms:
- Implicit form takes a plain memref/tensor as input to store; it doesn't
assume a particular buffering implementation (CBs, scratchpad, etc).

In tensor form (pre-bufferization), the op produces a result that aliases
the operand:

```mlir
// For an operand with device shape (GY x GX) x (SY x SX)
%result = d2m.remote_store %generic_operand[%i, %j] %buf
  : tensor<GY x GX x SY x SX x f32, ...>, tensor<SY x SX x f32, ...>
  -> tensor<GY x GX x SY x SX x f32, ...>
```

In memref form (post-bufferization), the op has no result:

```mlir
// For an operand with device shape (GY x GX) x (SY x SX)
d2m.remote_store %generic_operand[%i, %j] %buf
  : memref<GY x GX x SY x SX x f32, ...>, memref<SY x SX x f32, ...>
```

- Explicit CB form stores data from a concrete CB (from `d2m.get_cb`) to the remote operand.

```mlir
d2m.remote_store %generic_operand[%i, %j] from %cb
  : memref<GY x GX x SY x SX x f32, ...> from !d2m.cb<memref<SY x SX x f32, ...>>
```

In tensor form (pre-bufferization), the result matches the type of the
operand tensor, allowing it to be used as a yield value in a generic region
and forming a complete use-chain from remote_load to remote_store. In memref
form (post-bufferization), there is no result.

Constraints:
- The number of indices must equal N/2, where N is the memref/tensor rank.
- The operation stores the entire shard from a local buffer or circular buffer
- If in explicit CB form:
  - The operand provided must be a memref
  - The CB and operand memref must have compatible shard shapes
- Exactly one of localBuffer or cb must be present

## `d2m.reserve`

Reserve from circular buffer.

Reserve operation, extracts the enclosing memref from a circular buffer.
This operation is used by producer threads to reserve a memref sized underlying
type's worth of space in the circular buffer. It implicitly blocks if there hasn't
yet been a consumer of the data from another thread via d2m.wait.

Each value of !d2m.cb type can be thought of as a shared resource between threads
akin to a single-producer, single-consumer queue. Where d2m.reserve and d2m.wait
effectively implement push/pop queue semantics. One distinction which lends itself
better to DPS style is that both reserve and wait guarantee acquisition of the
underlying memory after the op has executed.

The resource is implicitly released at the end of block scope, OR explicitly
released via d2m.push.

Example:
```mlir
%memref = d2m.reserve %cb : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
```

## `d2m.reset_global_semaphore`

Global semaphore reset operation (D2M).

Reset global semaphore to a specific value. This operation blocks until the value has been set.

## `d2m.scratch_allocate`

Allocate scratch memory for intermediate spilling.

Allocate a scratch buffer from the L1 scratchpad region for storing
intermediate results.

The `slot` attribute is a unique identifier for this scratch allocation
within the enclosing scope, used for tracking and debugging.

This op is lowered by `LowerScratchAllocate` to a `memref.subview` into
the scratch memref anchored by `d2m.scratch_init`.

Example:
```mlir
%scratch = d2m.scratch_allocate {slot = 0 : i64}
           : memref<1x!ttcore.tile<32x32xbf16>, #ttcore.memory_space<l1>>
```

## `d2m.scratch_init`

Anchor op for scratch buffer memref.

Marks a `memref.alloc` as the scratch buffer for intermediate spilling
inside a `d2m.generic` region. This op consumes the memref to prevent
canonicalization from removing the otherwise-unused alloc.

Inserted by `InsertScratchBuffers` and erased by `LowerScratchAllocate`
after `scratch_allocate` ops are replaced with subviews of this memref.

Example:
```mlir
%scratch = memref.alloc() : memref<1x32x!ttcore.tile<32x32, f32>,
                                   #ttcore.memory_space<l1>>
d2m.scratch_init %scratch : memref<1x32x!ttcore.tile<32x32, f32>,
                                   #ttcore.memory_space<l1>>
```

## `d2m.semaphore_set`

D2M Semaphore Set Op.

Wait for the semaphore value to reach the specified value. Optionally supply a reset value as a shorthand syntax.

```mlir
d2m.semaphore_wait %sem1, %c1 reset %c0
// is equivalent to
d2m.semaphore_wait %sem1, %c1
d2m.semaphore_set %sem1, %c0
```

## `d2m.set_l1_accumulate`

Enable L1 accumulation mode.

Configures HW state such that all subsequent stores to L1 memory space
will be accumulated to in place.

## `d2m.spatial`

Execute multiple d2m.generic ops simultaneously on disjoint grid ranges of the device.

Execute multiple d2m.generic ops simultaneously on disjoint parts of the physical
device. Each region corresponds to one grid range in `grid_ranges` and contains
exactly one d2m.generic op. Grid ranges must be non-overlapping and contained
in the device.

## `d2m.spatial_yield`

Yield op for spatial op.

Yield operation for spatial op.

## `d2m.store`

TODO.

TODO

## `d2m.tile_abs`

D2M Tile Abs Op

The `tile_abs` operation computes the absolute value of each element in the input tile.

## `d2m.tile_acos`

D2M Tile Acos Op

The `tile_acos` operation computes the arccosine of each element in the input tile.

## `d2m.tile_add`

D2M Tile Add Op

The `tile_add` operation adds two tiles element-wise on the FPU (or SFPU if one of the operands is a float32),
or adds a scalar value to each element in the input tile if rhs is a scalar (SFPU path).

## `d2m.tile_argmax`

D2M Tile ArgMax Op

The `tile_argmax` operation computes the max of all elements in the input values along
with its index over the columns (i.e. the output is a row). Requires the input values
and indices to be in row-major layout. Indices must be properly filled in and passed
by the callee. The val_acc_out and idx_acc_out are the accumulator tiles that can be used
to reduce large reduction dims. Tie-breaking behavior is different from PyTorch, the LLK
picks the highest index in case of a tie.

## `d2m.tile_asin`

D2M Tile Asin Op

The `tile_asin` operation computes the arcsine of each element in the input tile.

## `d2m.tile_atan`

D2M Tile Atan Op

The `tile_atan` operation computes the arctangent of each element in the input tile.

## `d2m.tile_atan2`

D2M Tile Atan2 Op

The `tile_atan2` operation computes the element-wise arctangent of `lhs` over `rhs`.

## `d2m.tile_bcast`

D2M Tile Broadcast Op

The `tile_bcast` operation broadcasts a row/col/scalar tile into a full tile.

## `d2m.tile_bitwise_and`

D2M Tile Bitwise And Op

The `tile_bitwise_and` operation computes bitwise AND between two tiles.

## `d2m.tile_bitwise_not`

D2M Tile Bitwise Not Op

The `tile_bitwise_not` operation computes the bitwise negation of each element in the input tile.

## `d2m.tile_bitwise_or`

D2M Tile Bitwise Or Op

The `tile_bitwise_or` operation computes bitwise OR between two tiles.

## `d2m.tile_bitwise_xor`

D2M Tile Bitwise Xor Op

The `tile_bitwise_xor` operation computes bitwise XOR between two tiles.

## `d2m.tile_ceil`

D2M Tile Ceil Op

The `tile_ceil` operation computes the ceiling function of each element in the input tile.

## `d2m.tile_clamp_scalar`

D2M Tile Clamp Scalar Op

The `tile_clamp_scalar` operation clamps all elements of a tile to be within
the range [min, max] specified by scalar attributes.

## `d2m.tile_cos`

D2M Tile Cos Op

The `tile_cos` operation computes the cosine function of each element in the input tile.

## `d2m.tile_div`

D2M Tile Div Op

The `tile_div` operation divides two tiles element-wise,
or divides each element in the input tile by a scalar value if rhs is a scalar.

## `d2m.tile_eq`

D2M Tile Eq Op

The `tile_eq` operation performs element-wise equality comparison between two tiles.
Result element is 1.0 where lhs == rhs, otherwise 0.0.

## `d2m.tile_eqz`

D2M Tile Eqz Op

The `tile_eqz` operation checks if each element in the input tile == 0

## `d2m.tile_erf`

D2M Tile Erf Op

The `tile_erf` operation computes the error function (erf) of each element in the input tile.

## `d2m.tile_erfc`

D2M Tile Erfc Op

The `tile_erfc` operation computes the complementary error function (erfc) of each element in the input tile.

## `d2m.tile_exp`

D2M Tile Exp Op

The `tile_exp` operation computes the exponential of each element in the input tile.

## `d2m.tile_exp2`

D2M Tile Exp2 Op

The `tile_exp2` operation computes the base-2 exponential of each element in the input tile.

## `d2m.tile_expm1`

D2M Tile Expm1 Op

The `tile_expm1` operation computes exp(x)-1 of each element in the input tile.

## `d2m.tile_fill`

D2M Fill Tile Op

The `tile_fill` operation creates a tile filled with a constant scalar value.
All elements in the resulting tile will have the same value as the input scalar.

This operation abstracts away the complexity of creating a constant-filled tile,
providing a simple interface that takes a scalar value and produces a tile suitable
for element-wise operations.

## `d2m.tile_floor`

D2M Tile Floor Op

The `tile_floor` operation computes the floor function of each element in the input tile.

## `d2m.tile_frac`

D2M Tile Frac Op

The `tile_frac` operation computes the fractional part of each element in the input tile.

## `d2m.tile_ge`

D2M Tile Ge Op

The `tile_ge` operation performs element-wise greater-than-or-equal comparison
between two tiles. Result element is 1.0 where lhs >= rhs, otherwise 0.0.

## `d2m.tile_gelu`

D2M Tile GELU Op

The `tile_gelu` operation computes the GELU (Gaussian Error Linear Unit) of each element in the input tile.

## `d2m.tile_gez`

D2M Tile Gez Op

The `tile_gez` operation checks if each element in the input tile >= 0

## `d2m.tile_gt`

D2M Tile Gt Op

The `tile_gt` operation performs element-wise greater-than comparison between two tiles.
Result element is 1.0 where lhs > rhs, otherwise 0.0.

## `d2m.tile_gtz`

D2M Tile Gtz Op

The `tile_gtz` operation checks if each element in the input tile > 0

## `d2m.tile_hardsigmoid`

D2M Tile Sigmoid Op

The `tile_hardsigmoid` operation computes the hardsigmoid of each element in the input tile.

## `d2m.tile_le`

D2M Tile Le Op

The `tile_le` operation performs element-wise less-than-or-equal comparison
between two tiles. Result element is 1.0 where lhs <= rhs, otherwise 0.0.

## `d2m.tile_lez`

D2M Tile Lez Op

The `tile_lez` operation checks if each element in the input tile <= 0

## `d2m.tile_log`

D2M Tile Log Op

The `tile_log` operation computes the natural logarithm of each element in the input tile.

## `d2m.tile_log1p`

D2M Tile Log1p Op

The `tile_log1p` operation computes log(1+x) of each element in the input tile.

## `d2m.tile_logical_left_shift`

D2M Tile Logical Left Shift Op

The `tile_logical_left_shift` operation computes an elementwise logical
left shift of `lhs` by `rhs` bits between two integer tiles.

## `d2m.tile_logical_not`

D2M Tile Logical Not Op

The `tile_logical_not` operation computes the logical negation of each element in the input tile.

## `d2m.tile_logical_right_shift`

D2M Tile Logical Right Shift Op

The `tile_logical_right_shift` operation computes an elementwise logical
right shift of `lhs` by `rhs` bits between two integer tiles. Unlike an
arithmetic right shift, the sign bit is not propagated; the high-order
bits are filled with zeros.

## `d2m.tile_lt`

D2M Tile Lt Op

The `tile_lt` operation performs element-wise less-than comparison between two tiles.
Result element is 1.0 where lhs < rhs, otherwise 0.0.

## `d2m.tile_ltz`

D2M Tile Ltz Op

The `tile_ltz` operation checks if each element in the input tile < 0

## `d2m.tile_matmul`

D2M Tile Matmul Op

The `tile_matmul` operation computes the matrix multiplication of A and B input tiles and element-wise adds C tile: result <- a @ b + c.

## `d2m.tile_matmul_block`

D2M Tile Matmul Block Op

The `tile_matmul_block` operation computes the matrix multiplication of two input blocks.

The optional `transpose_b` attribute controls whether the B operand is
transposed during the matmul. This maps onto the `transpose` flag
supported natively by the underlying ttkernel matmul_block API.

## `d2m.tile_maximum`

D2M Tile Maximum Op

The `tile_maximum` operation calculates the maximum of two tensors element-wise.

## `d2m.tile_minimum`

D2M Tile Minimum Op

The `tile_minimum` operation calculates the minimum of two tensors element-wise.

## `d2m.tile_mul`

D2M Tile Mul Op

The `tile_mul` operation multiplies two tiles element-wise on the FPU (or SFPU if one of the operands is a float32),
or multiplies each element in the input tile by a scalar value if rhs is a scalar (SFPU path).

## `d2m.tile_ne`

D2M Tile Ne Op

The `tile_ne` operation performs element-wise inequality comparison between two tiles.
Result element is 1.0 where lhs != rhs, otherwise 0.0.

## `d2m.tile_negative`

D2M Tile Negative Op

The `tile_negative` operation computes the negative of each element in the input tile.

## `d2m.tile_nez`

D2M Tile Nez Op

The `tile_nez` operation checks if each element in the input tile != 0

## `d2m.tile_pow`

D2M Tile Pow Op

The `tile_pow` operation raises two tiles element-wise,
or raises each element in the input tile to a scalar power if rhs is a scalar.

## `d2m.tile_rand`

D2M Rand Tile Op

The `tile_rand` operation creates a tile filled with random float values
drawn from a uniform distribution over the range [from, from + scale).

Attributes:
  - `seed` (UI32): Seed passed to `rand_tile_init` to initialize the PRNG.
  - `from` (F32): Lower bound of the uniform distribution (inclusive).
  - `scale` (F32): Width of the uniform distribution (must be > 0).

Lowers to `ttkernel.rand_tile_init(seed)` + `ttkernel.rand_tile(dst_idx,
from_bits, scale_bits)` where the `from`/`scale` parameters are the
bit-cast f32 values.

## `d2m.tile_recip`

D2M Tile Recip Op

The `tile_recip` operation computes the reciprocal of each element in the input tile.

## `d2m.tile_reduce_max`

D2M Tile Reduce Max Op

The `tile_reduce_max` operation computes the max of all elements in the input A element-wise multiplied by B and input C over the specified reduction dim(s): result <- max<dims>(A * B, C).
Valid only for float element types; integer max reductions use `tile_sfpu_reduce_max` instead.

## `d2m.tile_reduce_mean`

D2M Tile Reduce Mean Op

The `tile_reduce_mean` operation computes the mean of all elements in the input A element-wise multiplied by B and input C over the specified reduction dim(s): result <- avg<dims>(A * B, C).
Valid only for float element types; integer mean reduction is not supported yet.

## `d2m.tile_reduce_sum`

D2M Tile Reduce Sum Op

The `tile_reduce_sum` operation computes the sum of all elements in the input A element-wise multiplied by B and input C over the specified reduction dim(s): result <- sum<dims>(A * B, C).
Valid only for float element types; integer sum reductions use `tile_sfpu_reduce_sum` instead.

## `d2m.tile_relu`

D2M Tile Relu Op

The `tile_relu` operation computes the relu of each element in the input tile.

## `d2m.tile_right_shift`

D2M Tile Arithmetic Right Shift Op

The `tile_right_shift` operation computes an elementwise arithmetic
right shift of `lhs` by `rhs` bits between two integer tiles. The
sign bit is propagated for signed integer element types.

## `d2m.tile_rsqrt`

D2M Tile Rsqrt Op

The `tile_rsqrt` operation computes the reciprocal sqrt function of each element in the input tile.

## `d2m.tile_selu`

D2M Tile SELU Op

The `tile_selu` operation applies SELU activation with `scale` and `alpha` given as
f32 attributes (passed to the kernel as IEEE-754 bit patterns).

## `d2m.tile_sfpu_reduce_max`

D2M Tile SFPU Reduce Max Op

The `tile_sfpu_reduce_max` operation computes the max of the input A over the specified reduction dim(s), accumulating into C: result <- max<dims>(A, C).
Valid only for integer element types; float max reductions use `tile_reduce_max` instead.

`dst_scratch_index` is populated by `d2m-insert-dst-register-access`
(see `getNumDstScratchSlices`) and is -1 before the pass runs.

## `d2m.tile_sfpu_reduce_sum`

D2M Tile SFPU Reduce Sum Op

The `tile_sfpu_reduce_sum` operation computes the sum of the input A over the specified reduction dim(s), accumulating into C: result <- sum<dims>(A, C).
Valid only for integer element types; float sum reductions use `tile_reduce_sum` instead.

`dst_scratch_index` is populated by `d2m-insert-dst-register-access`
(see `getNumDstScratchSlices`) and is -1 before the pass runs.

## `d2m.tile_sigmoid`

D2M Tile Sigmoid Op

The `tile_sigmoid` operation computes the sigmoid of each element in the input tile.

## `d2m.tile_sign`

D2M Tile Sign Op

The `tile_sign` operation computes the sign of each element in the input tile.
Returns 1 for positive values, -1 for negative values, and 0 for zero.

## `d2m.tile_signbit`

D2M Tile Signbit Op

The `tile_signbit` operation extracts the IEEE-754 sign bit of each element,
producing 0.0 or 1.0 in the tile element type.

## `d2m.tile_silu`

D2M Tile Sliu Op

The `tile_silu` operation computes the silu of each element in the input tile.

## `d2m.tile_sin`

D2M Tile Sin Op

The `tile_sin` operation computes the sine function of each element in the input tile.

## `d2m.tile_softsign`

D2M Tile Softsign Op

The `tile_softsign` operation computes x/(1+|x|) for each element in the input tile.

## `d2m.tile_sqrt`

D2M Tile Sqrt Op

The `tile_sqrt` operation computes the sqrt function of each element in the input tile.

## `d2m.tile_square`

D2M Tile Square Op

The `tile_square` operation computes the square function of each element in the input tile.

## `d2m.tile_sub`

D2M Tile Sub Op

The `tile_sub` operation subtracts two tiles element-wise on the FPU (or SFPU if one of the operands is a float32),
or subtracts a scalar value from each element in the input tile if rhs is a scalar (SFPU path).

## `d2m.tile_tan`

D2M Tile Tan Op

The `tile_tan` operation computes the tangent function of each element in the input tile.

## `d2m.tile_tanh`

D2M Tile Tanh Op

The `tile_tanh` operation computes the hyperbolic tangent function of each element in the input tile.

## `d2m.tile_tilize_block`

D2M Tile Tilize Block Op

The `tile_tilize_block` operation tilizes the input row major memref block and outputs the memref containing the tilized data.

## `d2m.tile_topk_local_sort`

D2M Tile TopK Local Sort Block Op

Runs a bitonic sort in place over the concatenated 2-tile sequence
formed by `tile_a` and `tile_b` (each a value tile paired with an
index tile), leaving each tile as a sorted run forming one bitonic
sequence across both. Maps onto the ttkernel `topk_local_sort` LLK
call.

- `values`/`indices`: input value/index tensors or memrefs.
- `out_values`/`out_indices`: output value/index tensors or memrefs;
  the op's DPS inits (`getDpsInitsMutable` = operands 2-3).
- `idir`: sort direction (0 = descending).
- `i_end_phase`: last bitonic-network phase (0-indexed) to run; a full
  64-element sort runs phases 0..5.
- `i_start_phase`: first bitonic-network phase (0-indexed) to run;
  nonzero values resume a partially-sorted tile pair rather than
  re-sorting from scratch.
- `tile_a`, `tile_b`: flat tile indices of the pair being sorted.
- `is_group_start`: whether this op is responsible for acquiring the
  DST tile registers and copying `tile_a`/`tile_b` (values and
  indices) in from the source CB.
- `is_group_end`: whether this op is responsible for packing the
  resulting DST tiles back to the output CB and releasing the DST
  tile registers.
- `read_from_output`: whether `tile_a`/`tile_b` are read from the
  output value/index buffers instead of the input buffers.

## `d2m.tile_topk_merge`

D2M Tile TopK Merge Block Op

Given two bitonic length-K subsequences held in `tile_a` and
`tile_b`, merges them so the top-K values/indices across both land in
`tile_a`; `tile_b` is left holding the complementary "losers". Maps
onto the ttkernel `topk_merge` LLK call.

- `values`/`indices`: input value/index tensors or memrefs.
- `out_values`/`out_indices`: output value/index tensors or memrefs;
  the op's DPS inits (`getDpsInitsMutable` = operands 2-3).
- `m_iter`: merge-tree level (0-indexed) being merged; determines
  which stride/shape of length-K runs the LLK merges.
- `k`: number of top elements tracked per subsequence.
- `tile_a`, `tile_b`: flat tile indices of the pair being merged;
  after the merge, `tile_a` holds the combined top-K result, `tile_b`
  the losers.
- `is_group_start`: whether this op is responsible for acquiring the
  DST tile registers and copying `tile_a`/`tile_b` in from the source
  CB.
- `is_group_end`: whether this op is responsible for packing the
  resulting DST tiles back to the output CB and releasing the DST
  tile registers.
- `read_from_output`: whether `tile_a`/`tile_b` are read from the
  output value/index buffers instead of the input buffers.

## `d2m.tile_topk_rebuild`

D2M Tile TopK Rebuild Block Op

Re-sorts the length-K sequence held across `tile_a`/`tile_b` into
fully-ordered order (a merge only guarantees the top-K values landed
in `tile_a`, not that they're sorted). Maps onto the ttkernel
`topk_rebuild` LLK call.

- `values`/`indices`: input value/index tensors or memrefs.
- `out_values`/`out_indices`: output value/index tensors or memrefs;
  the op's DPS inits (`getDpsInitsMutable` = operands 2-3).
- `idir`: sort direction (0 = descending).
- `m_iter`: merge-tree level (0-indexed) whose result is being
  rebuilt.
- `k`: number of top elements retained in the rebuilt sequence.
- `logk`: `floor(log2(k))`; number of bitonic-network phases the
  rebuild runs to fully re-sort the length-K sequence.
- `skip_second`: whether the rebuild's second sub-pass (over the
  second half of the sequence) is skipped.
- `tile_a`, `tile_b`: flat tile indices of the pair being rebuilt;
  after rebuild hold the fully sorted top-K result.
- `is_group_start`: whether this op is responsible for acquiring the
  DST tile registers and copying `tile_a`/`tile_b` in from the source
  CB.
- `is_group_end`: whether this op is responsible for packing the
  resulting DST tiles back to the output CB and releasing the DST
  tile registers.
- `read_from_output`: whether `tile_a`/`tile_b` are read from the
  output value/index buffers instead of the input buffers.

## `d2m.tile_transpose`

D2M Tile Transpose Op

The `tile_transpose` operation computes the transpose of the input tile.

## `d2m.tile_trunc`

D2M Tile Trunc Op

The `tile_trunc` operation truncates each element of the input tile toward zero.

## `d2m.tile_typecast`

D2M Tile Typecast Op

The `tile_typecast` operation casts the input tile to the desired dataformat.

## `d2m.tile_untilize_block`

D2M Tile Untilize Block Op

The `tile_untilize_block` operation untilizes the input tilized memref block and outputs the memref containing the row major data.

## `d2m.tile_where`

D2M Tile Where Op

The `tile_where` operation performs element-wise conditional selection.
For each element position, it selects between two values based on a boolean
condition in the first tensor:
- If the condition is true (non-zero), it selects the corresponding element
  from the second tensor (true_value)
- If the condition is false (zero), it selects the corresponding element
  from the third tensor (false_value)

## `d2m.topk_block`

D2M TopK Block Op

The `topk_block` operation performs the core TopK computation on a
transposed input shard, producing sorted top-K values and their
original indices.

This op operates at the block level (memref of tiles) and is
decomposed post-bufferization by `d2m-decompose-topk` into
`tile_topk_local_sort`, `tile_topk_merge`, and `tile_topk_rebuild`.

`generate_indices` selects where the index buffer paired with the value
tiles comes from:

- false (the default): scratch_idx_tile already holds one index tile
  per value tile. Used by merge stages, whose indices can't be
  recomputed.
- true: scratch_idx_tile is just a 1x1 scratch tile, and
  `d2m-decompose-topk` builds the index buffer in-kernel from the
  core's grid coordinate. Used by a leaf topk over raw input.

## `d2m.unpack_stall_on_pack`

Stall UNPACK thread until previous PACK write is committed to L1.

Inserted between consecutive linalg.generics in a fused region when
the intermediate result is written to L1 by one generic and immediately
read by the next, and the usual tile_regs_acquire semwait is not present
(single-tile shard path).

## `d2m.wait`

Wait from circular buffer.

Wait operation, extracts the enclosing memref from a circular buffer.
This operation is used by consumer threads to access data from the circular buffer.
It implicitly blocks until a chunk of memref sized underlying type is made
available by a producer thread via d2m.reserve.

Each value of !d2m.cb type can be thought of as a shared resource between threads
akin to a single-producer, single-consumer queue. Where d2m.reserve and d2m.wait
effectively implement push/pop queue semantics. One distinction which lends itself
better to DPS style is that both reserve and wait guarantee acquisition of the
underlying memory after the op has executed.

The resource is implicitly released at the end of block scope, OR explicitly
released via d2m.pop.

Example:
```mlir
%memref = d2m.wait %cb : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
```

## `d2m.write_col_mask_tile`

Write column mask pattern to a tile CB

The `write_col_mask_tile` operation writes a column mask pattern
to a tile CB, where element[i,j] = 1.0 if j < validCols, else 0.0.

This is used for partial tile OOB masking. The operation writes directly
to L1 memory, avoiding DST register pressure.

## `d2m.write_row_mask_tile`

Write row mask pattern to a tile CB

The `write_row_mask_tile` operation writes a row mask pattern
to a tile CB, where element[i,j] = 1.0 if i < validRows, else 0.0.

This is used for partial tile OOB masking. The operation writes directly
to L1 memory, avoiding DST register pressure.

## `d2m.yield`

Yield op.

D2M yield equivalent, required for enforcing pure tensor semantics on the tensor form of the GenericOp.

