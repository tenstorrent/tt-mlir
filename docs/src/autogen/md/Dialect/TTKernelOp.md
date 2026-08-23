# TTKernel Dialect Ops

Auto-generated reference of operations in the `TTKernel` dialect (292 ops).

## `ttkernel.TensorAccessor`

MakeTensorAccessorFromArgs

TensorAccessor constructor.

## `ttkernel.TensorAccessorArgs`

TensorAccessorArgs

TensorAccessorArgs struct constructor.

CTA (compile-time args) and CRTA (compile/runtime args) offsets are determined as follows.
1. If cta_expr is provided: use the constexpr string expression.
2. Else if `prev_args` is provided, use chaining:
  `prev_args.next_compile_time_args_offset()` (for CTA) or
  `prev_args.next_common_runtime_args_offset()` (for CRTA).
3. Otherwise, use cta_base/crta_base integer constants.

Examples:
```mlir
// Literal offsets
%c0 = arith.constant 0 : i32
%args = ttkernel.TensorAccessorArgs(%c0, %c0) : (i32, i32) -> !ttkernel.TensorAccessorArgs
// Generates: TensorAccessorArgs<0, 0>()

// CTA+CRTA chaining (common pattern)
%args_src = ttkernel.TensorAccessorArgs(%c0, %c0) : (i32, i32) -> !ttkernel.TensorAccessorArgs
%args_dst = ttkernel.TensorAccessorArgs(prev = %args_src) : (!ttkernel.TensorAccessorArgs) -> !ttkernel.TensorAccessorArgs
// Generates: TensorAccessorArgs<args_src.next_compile_time_args_offset(), args_src.next_common_runtime_args_offset()>()

// Selective override: chain CTA from prev, use literal CRTA
%args_custom = ttkernel.TensorAccessorArgs(prev = %args_src) {crta_expr = "0"} : (!ttkernel.TensorAccessorArgs) -> !ttkernel.TensorAccessorArgs
// Generates: TensorAccessorArgs<args_src.next_compile_time_args_offset(), 0>()

// Constexpr expression (no chaining)
%args_expr = ttkernel.TensorAccessorArgs(%c0, %c0) {cta_expr = "get_offset()"} : (i32, i32) -> !ttkernel.TensorAccessorArgs
// Generates: TensorAccessorArgs<get_offset(), 0>()
```

## `ttkernel.abs_tile`

Absolute value tile in the DST at specified index.

Performs element-wise computation of absolute value operation
DST[dst0_index] <- abs(DST[dst0_index])
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.

## `ttkernel.abs_tile_init`

Init function for abs_tile operation. Refer to documentation for any init function.

Must be run before abs_tile.

## `ttkernel.abs_tile_int32`

Absolute value i32 tile in the DST at specified index.

Performs element-wise computation of absolute value operation
DST[dst0_index] <- abs(DST[dst0_index])
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.

## `ttkernel.acos_tile`

Arccosine tile in the DST at specified index.

Performs element-wise computation of arccosine operation
DST[dst0_index] <- acos(DST[dst0_index])
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.

## `ttkernel.acos_tile_init`

Init function for acos_tile operation. Refer to documentation for any init function.

Must be run before acos_tile.

## `ttkernel.add_binary_tile`

Addition operation between two tiles

Performs element-wise computation of addition operation
DST[odst_index] <- DST[dst0_index] + DST[dst1_index]
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.

## `ttkernel.add_binary_tile_init`

Short init function

Must be run before add_binary_tile.

## `ttkernel.add_int_tile`

Integer addition operation between two tiles

Performs element-wise computation of integer addition operation
DST[odst_index] <- DST[dst0_index] + DST[dst1_index]
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.
Supported data formats are: Int32, UInt32, UInt16.

## `ttkernel.add_int_tile_init`

Short init function

Must be run before add_int_tile.

## `ttkernel.add_tiles`

Add operation

Performs element-wise addition C=A+B of tiles in two CBs at given indices
and writes the result to the DST register at index dst_tile_index. The DST
register buffer must be in acquired state via *tile_regs_acquire* call. This call
is blocking and is only available on the compute engine.

## `ttkernel.add_tiles_init`

Short init function

Must be run before add_tiles.

## `ttkernel.add_unary_tile`

Add by scalar operation

Performs element-wise addition of a tile by a scalar value.
DST[dst0_index] <- DST[dst0_index] + scalar
The DST register buffer must be in acquired state via *tile_regs_acquire* call.
This call is blocking and is only available on the compute engine.

## `ttkernel.add_unary_tile_int32`

Add by int32 scalar operation

Performs element-wise addition of an int32 tile by an int32 scalar value.
DST[dst0_index] <- DST[dst0_index] + scalar
The DST register buffer must be in acquired state via *tile_regs_acquire* call.
This call is blocking and is only available on the compute engine.

## `ttkernel.asin_tile`

Arcsine tile in the DST at specified index.

Performs element-wise computation of arcsine operation
DST[dst0_index] <- asin(DST[dst0_index])
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.

## `ttkernel.asin_tile_init`

Init function for asin_tile operation. Refer to documentation for any init function.

Must be run before asin_tile.

## `ttkernel.atan2_binary_tile`

Elementwise atan2 operation between two tiles

Performs element-wise computation of atan2 operation
DST[odst_index] <- atan2(DST[dst0_index], DST[dst1_index])
on DST register operands. The DST register buffer must be in
acquired state via *acquire_dst* call.

## `ttkernel.atan2_binary_tile_init`

Short init function

Must be run before atan2_binary_tile.

## `ttkernel.atan_tile`

Arctangent tile in the DST at specified index.

Performs element-wise computation of arctangent operation
DST[dst0_index] <- atan(DST[dst0_index])
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.

## `ttkernel.atan_tile_init`

Init function for atan_tile operation. Refer to documentation for any init function.

Must be run before atan_tile.

## `ttkernel.bfloat16_greater`

Compare two bfloat16 values using integer arithmetic.

Returns true if bf16_a > bf16_b.  Operates on raw bits in int16 format
representation.  Maps to bfloat16_greater() in device code.

## `ttkernel.binary_bitwise_tile_init`

Init function for binary bitwise operations (AND, OR, XOR). Refer to documentation for any init function.

Must be run before bitwise_and_binary_tile, bitwise_or_binary_tile, or bitwise_xor_binary_tile.

## `ttkernel.binary_dest_reuse_tiles`

Binary op with one operand from DST

Performs element-wise binary op where one operand comes from DST and one
from a CB. If reuse_type is dest_to_srca, DST[dst_tile_index] is loaded
to SRCA and CB tile is loaded to SRCB. If dest_to_srcb, the opposite.
Result is written back to DST[dst_tile_index].
The DST register buffer must be in acquired state via *tile_regs_acquire* call.
`eltwise_binary_type` specifies the operation (add/sub/mul).

## `ttkernel.binary_dest_reuse_tiles_init`

Init for binary op with dest reuse

Init function for binary_dest_reuse_tiles operation.
Must be run before binary_dest_reuse_tiles.
`eltwise_binary_type` specifies the operation (add/sub/mul).
`reuse_type` specifies which source register gets the DST operand.

## `ttkernel.binary_left_shift_tile`

Elementwise left shift between two tiles

Performs element-wise computation of a left shift operation
DST[odst_index] <- DST[dst0_index] << DST[dst1_index]
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.

## `ttkernel.binary_logical_right_shift_tile`

Elementwise logical right shift between two tiles

Performs element-wise computation of a logical right shift operation
DST[odst_index] <- DST[dst0_index] >> DST[dst1_index] (zero-filled)
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.

## `ttkernel.binary_max_int32_tile`

Integer elementwise maximum operation between two tiles

Performs element-wise computation of integer maximum operation
DST[odst_index] <- max(DST[dst0_index], DST[dst1_index])
on DST register operands. The DST register buffer must be in
acquired state via *acquire_dst* call.

## `ttkernel.binary_max_int32_tile_init`

Short init function

Must be run before binary_max_int32_tile.

## `ttkernel.binary_max_tile`

Elementwise maximum operation

Performs element-wise computation of maximum operation
DST[odst_index] <- max(DST[dst0_index], DST[dst1_index])
on DST register operands. The DST register buffer must be in
acquired state via *acquire_dst* call.

## `ttkernel.binary_max_tile_init`

Short init function

Must be run before binary_max_tile.

## `ttkernel.binary_min_int32_tile`

Integer elementwise minimum operation between two tiles

Performs element-wise computation of integer minimum operation
DST[odst_index] <- min(DST[dst0_index], DST[dst1_index])
on DST register operands. The DST register buffer must be in
acquired state via *acquire_dst* call.

## `ttkernel.binary_min_int32_tile_init`

Short init function

Must be run before binary_min_int32_tile.

## `ttkernel.binary_min_tile`

Elementwise minimum operation

Performs element-wise computation of minimum operation
DST[odst_index] <- min(DST[dst0_index], DST[dst1_index])
on DST register operands. The DST register buffer must be in
acquired state via *acquire_dst* call.

## `ttkernel.binary_min_tile_init`

Short init function

Must be run before binary_min_tile.

## `ttkernel.binary_op_init_common`

Init function for all binary ops

Followed by the specific init required with an opcode (binrary_op_specific_init).

## `ttkernel.binary_right_shift_tile`

Elementwise arithmetic right shift between two tiles

Performs element-wise computation of an arithmetic right shift
operation DST[odst_index] <- DST[dst0_index] >> DST[dst1_index]
(sign-preserving) on DST register operands. The DST register buffer must
be in acquired state via *tile_regs_acquire* call.

## `ttkernel.binary_shift_tile_init`

Init function for binary shift tile operations (left / right / logical right). Refer to documentation for any init function.

Must be run before binary_left_shift_tile, binary_right_shift_tile, or binary_logical_right_shift_tile.

## `ttkernel.binop_with_scalar_tile_init`

Init function for binary operations with scalar tile operations.

Must be run before binary operations with scalar like mul_unary_tile.

## `ttkernel.bitcast`

Reinterpret a ui32 compile-time arg value as a scalar type.

Reinterprets the bits of a ui32 compile-time arg as the given type.
Used to recover typed scalar kernel arguments after reading them from
the compile-time arg slot (which always stores ui32).

## `ttkernel.bitwise_and_binary_tile`

Bitwise AND operation between two tiles

Performs element-wise computation of bitwise AND operation
DST[odst_index] <- DST[dst0_index] & DST[dst1_index]
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.

## `ttkernel.bitwise_not_tile`

Bitwise Not operation on tile in the DST at specified index.

Performs element-wise computation of sign operation
DST[tile_index] <- bitwise_not(DST[tile_index])
on DST register operands.
The DST register buffer must be in acquired state via *tile_regs_acquire* call.
This call is blocking and is only available on the compute engine.

## `ttkernel.bitwise_not_tile_init`

Init function for bitwise_not_tile operation. Refer to documentation for any init function.

Must be run before bitwise_not_tile.

## `ttkernel.bitwise_or_binary_tile`

Bitwise OR operation between two tiles

Performs element-wise computation of bitwise OR operation
DST[odst_index] <- DST[dst0_index] | DST[dst1_index]
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.

## `ttkernel.bitwise_xor_binary_tile`

Bitwise XOR operation between two tiles

Performs element-wise computation of bitwise XOR operation
DST[odst_index] <- DST[dst0_index] ^ DST[dst1_index]
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.

## `ttkernel.cb_pop_front`

CBPopFront call.

CBPopFront operation

## `ttkernel.cb_push_back`

CBPushBack call.

CBPushBack operation

## `ttkernel.cb_reserve_back`

CBReserveBack call.

CBReserveBack operation

## `ttkernel.cb_wait_front`

CBWaitFront call.

CBWaitFront operation

## `ttkernel.ceil_tile`

Ceil tile in the DST at specified index.

Performs element-wise computation of ceil operation
DST[dst0_index] <- ceil(DST[dst0_index])
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.

## `ttkernel.clamp_tile`

Clamp tile elements to scalar range

Performs element-wise clamping of tile values to the range [min, max]
DST[dst_index] <- clamp(DST[dst_index], min, max)
on DST register operand. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.
min and max are uint 32-bit representations of the float values.

## `ttkernel.clamp_tile_init`

Init function for clamp_scalar_tile operation.

Must be run before clamp_scalar_tile.

## `ttkernel.clamp_tile_int32`

Clamp int32 tile elements to scalar range

Performs element-wise clamping of int32 tile values to the range [min, max]
DST[dst_index] <- clamp(DST[dst_index], min, max)
on DST register operand. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.
min and max are int32 values.

## `ttkernel.compute_kernel_hw_startup`

compute_kernel_hw_startup

Must be run at the start of compute kernel.

## `ttkernel.copy_block_matmul_partials`

CopyBlockMatmulPartials op.

Copies ntiles consecutive tiles from a specified circular buffer starting
at start_tile_index to consecutive DST register slots starting at
start_dst_index. The function employs the unpacker to first unpack into
SRC registers and then move into DST registers.

The DST register buffer must be in acquired state via *tile_regs_acquire* call.
This call is blocking and is only available on the compute engine.

A typical use case is reloading partial matmul results from a CB back
into DST for further accumulation. The consumer first calls cb_wait_front
to ensure tiles are available, then calls copy_block_matmul_partials to
load the block into DST, accumulates with subsequent matmul operations,
and finally packs the results back to a CB.

## `ttkernel.copy_dest_values`

Copies all values from the tile at idst_in to the tile at idst_out in the DST register buffer.

Copies all values from the tile at idst_in to the tile at idst_out in the DST
register buffer. Performs element-wise copy: DST[idst_out] <- DST[idst_in].
The DST register buffer must be in acquired state via *tile_regs_acquire* call.
This call is blocking and is only available on the compute engine.

Lowers to: copy_dest_values<DataFormat>(idst_in, idst_out).

## `ttkernel.copy_dest_values_init`

Short init function

Must be run before copy_dest_values.

## `ttkernel.copy_tile`

Copy tile from specified CB to DST.

Copies a single tile from the specified input CB and writes the result to
DST at a specified index. The function will employ unpacker to first unpack into SRC
registers and then perform move into DST registers, at a specified index.
For the in_tile_index to be valid for this call, cb_wait_front(n) had to be
previously called to ensure that at least some number n>0 of tiles are available
in the input CB. The CB index 0 then references the first tile in the received section of the CB,
up to index n-1 (in a FIFO order). The DST register buffer must be in acquired state via
tile_regs_acquire call. This call is blocking and is only available on the compute
engine.

## `ttkernel.copy_tile_init`

Perform the init for copy tile. This does not reconfigure the unpacker data types.

Must be called before copy_tile.

## `ttkernel.cos_tile`

Cos operation

Performs element-wise computation of the trigonometric cosine operation on each element of a tile
in DST register at index tile_index. The DST register buffer must be in
acquired state via *acquire_dst* call. This call is blocking and is only
available on the compute engine.

## `ttkernel.cos_tile_init`

Short init function which configures compute unit for execution of cos_tile.

Must be run before cos_tile.

## `ttkernel.div_binary_tile`

Divide operation between two tiles

Performs element-wise computation of division operation
DST[odst_index] <- DST[dst0_index] / DST[dst1_index]
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.

## `ttkernel.div_binary_tile_init`

Short init function

Must be run before div_binary_tile.

## `ttkernel.div_unary_tile`

Divide by scalar operation

Performs element-wise division of a tile by a scalar value.
DST[dst0_index] <- DST[dst0_index] / scalar
The DST register buffer must be in acquired state via *tile_regs_acquire* call.
This call is blocking and is only available on the compute engine.

## `ttkernel.dprint`

Print to output stream from kernel.

std::format style format string:
```c++
rewriter.create<ttkernel::DPrintOp>(loc, "nocY={} nocX={} addr={}\\n",
                                  nocY, nocX, addr);
```
```
ttkernel.dprint("virtY {} virtX {} addr {}\\n", %14, %15, %13) : (index, index, i32)
```
Notes:
  - Only trivial format specifier currently supported, i.e. `{}`.
  - Must double escape newline character or other special characters.
  - When a CB operand is provided, calls print_cb_details, printing underlying CB pointers and details.

## `ttkernel.eq_binary_tile`

Equality operation between two tiles

Performs element-wise computation of equality operation
DST[odst_index] <- DST[dst0_index] == DST[dst1_index]
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.

## `ttkernel.eq_binary_tile_init`

Short init function

Must be run before eq_binary_tile.

## `ttkernel.eqz_tile`

Equal to zero tile in the DST at specified index.

Performs element-wise equality on DST register tiles.
DST[dst0_index] <- (DST[dst0_index] == 0)
The DST register buffer must be in acquired state via *tile_regs_acquire* call.
This call is blocking and is only available on the compute engine.

## `ttkernel.eqz_tile_init`

Init function for eqz() operation. Refer to documentation for any init function.

Must be run before eqz_tile.

## `ttkernel.eqz_tile_int32`

Equal to zero tile in the DST at specified index.

Performs element-wise equality on DST register tiles for int32 data type.
DST[dst0_index] <- (DST[dst0_index] == 0)
The DST register buffer must be in acquired state via *tile_regs_acquire* call.
This call is blocking and is only available on the compute engine.

## `ttkernel.erf_tile`

Erf operation

Performs element-wise computation of error function (erf) on each element of a tile
in DST register at index tile_index. The DST register buffer must be in
acquired state via *acquire_dst* call. This call is blocking and is only
available on the compute engine.

## `ttkernel.erf_tile_init`

Short init function which configures compute unit for execution of erf_tile.

Must be run before erf_tile.

## `ttkernel.erfc_tile`

Erfc operation

Performs element-wise computation of complementary error function (erfc) on each element of a tile
in DST register at index tile_index. The DST register buffer must be in
acquired state via *acquire_dst* call. This call is blocking and is only
available on the compute engine.

## `ttkernel.erfc_tile_init`

Short init function which configures compute unit for execution of erfc_tile.

Must be run before erfc_tile.

## `ttkernel.exp2_tile`

Exp2 operation

Performs element-wise computation of base-2 exponential (2^x) on each
element of a tile in DST register at index tile_index. The DST register
buffer must be in acquired state via *tile_regs_acquire* call. This call
is blocking and is only available on the compute engine.

## `ttkernel.exp2_tile_init`

Short init function which configures compute unit for execution of exp2_tile.

Must be run before exp2_tile.

## `ttkernel.exp_tile`

Exp operation

Performs element-wise computation of exponential on each element of a tile
in DST register at index tile_index. The DST register buffer must be in
acquired state via *tile_regs_acquire* call. This call is blocking and is only
available on the compute engine.

The optional attributes map onto the metal template parameters
`exp_tile<bool approx, bool scale_en, InputClamping input_clamping, int iterations>(idst, vector_mode, scale)`.
When all attributes are omitted the op lowers to a bare `exp_tile(idst)`
call, preserving the metal defaults
(approx=false, scale_en=false, input_clamping=ClampToNegative, iterations=8).
`scale` holds the runtime scale argument; when present and not equal to
the default 1.0 value it enables metal's `scale_en` template parameter and
is emitted as the third positional argument with `vector_mode` defaulted
to `VectorMode::RC`.

## `ttkernel.exp_tile_init`

Short init function which configures compute unit for execution of exp_tile.

Must be run before exp_tile.

The optional attributes map onto the metal template parameters
`exp_tile_init<bool approx, uint32_t scale, InputClamping input_clamping>()`.
When all attributes are omitted the op lowers to a bare `exp_tile_init()`
call, preserving the metal defaults
(approx=false, scale=0x3F800000, input_clamping=ClampToNegative).

## `ttkernel.experimental.close_fabric_connections`

CloseFabricConnections

Close fabric connections.

## `ttkernel.experimental.convert_logical_x_to_translated`

ConvertLogicalToTranslatedX

this converts the x coordinate from the LOGICAL coordinate system to TRANSLATED

## `ttkernel.experimental.convert_logical_y_to_translated`

ConvertLogicalToTranslatedY

this converts the y coordinate from the LOGICAL coordinate system to TRANSLATED

## `ttkernel.experimental.create_fabric_connection_manager`

CreateFabricConnectionManager

Create fabric connection manager. The fabric connection manager is required for all fabric operations.

## `ttkernel.experimental.fabric_fast_write_any_len`

FabricWriteOp

FabricWriteOp

## `ttkernel.experimental.fabric_mcast_fast_write_any_len`

FabricMulticastWriteOp

FabricMulticastWriteOp

## `ttkernel.experimental.fabric_mcast_sem_inc`

FabricMulticastSemIncOp

FabricMulticastSemIncOp. This operation increments a semaphore on a range of remote devices.

## `ttkernel.experimental.fabric_sem_inc`

FabricSemIncOp

FabricSemIncOp. This operation increments a semaphore on a remote device.

## `ttkernel.experimental.fill_arange_tile`

Experimental Write Full Linear Index Tile Op

Writes a full linear index tile pattern to a CB, where element[i,j] = i * 32 + j
(0-1023) in the CB's native data format (Float32, Float16_b, or Int32).

The resulting tile looks like:
[[  0,   1,   2, ...,  31],
 [ 32,  33,  34, ...,  63],
 ...
 [992, 993, 994, ..., 1023]]

## `ttkernel.experimental.get_device_id_from_logical_mesh_position`

GetDeviceIdFromLogicalMeshPosition

Get the device ID for a given logical mesh position.
Takes a list of indices representing the logical mesh position and returns the corresponding device ID.
Maps to get_device_id_from_logical_mesh_position(fcm, position) in device code.

## `ttkernel.experimental.get_my_device_id`

GetMyDeviceId

Get my device id. This is a 16 bit value.

## `ttkernel.experimental.get_my_logical_mesh_position`

GetMyLogicalMeshPosition

Get the mesh position for the current device at the given dimension.

## `ttkernel.experimental.matmul_block`

Matmul tiles operation

Performs block-sized matrix multiplication *C=A\*B* between the blocks in two
different input CBs and writes the result to DST. The DST register buffer
must be in acquired state via *acquire_dst* call. `in1_k_stride` is the
source-tile stride between consecutive K tiles in input 1. This call is
blocking and is only available on the compute engine.

## `ttkernel.experimental.pack_untilize_block`

Experimental PackUntilizeBlockOp call.

Custom pack untilize block LLK that takes the dimensions of the block.
Uses `pack_untilize_init` for initialization and calls
`pack_untilize_block<cols_per_dst_pass, total_col_tiles>(icb, ocb, block_r, block_c)`.
`cols_per_dst_pass` is the number of column tiles processed per DST pass
(constrained by DST capacity), and `total_col_tiles` is the total number
of column tiles in the untilized row.
For correctness, `cols_per_dst_pass` must divide `total_col_tiles`, and
`block_c` is expected to be compatible with the chosen
`cols_per_dst_pass` (the implementation processes
`block_c / cols_per_dst_pass` column blocks).
Use this op in the sequence:
`pack_untilize_init -> experimental.pack_untilize_block -> pack_untilize_uninit`.

## `ttkernel.experimental.semaphore_wait`

SemaphoreWait

A blocking call that waits until the value of a local L1 memory address on
the Tensix core executing this function becomes equal to a target value.
This L1 memory address is used as a semaphore of size 4 Bytes, as a
synchronization mechanism. Also, see *noc_semaphore_set*.

## `ttkernel.experimental.semaphore_wait_min`

SemaphoreWaitMin

A blocking call that waits until the value of a local L1 memory address on
the Tensix core executing this function becomes equal or greater than a target value.
This L1 memory address is used as a semaphore of size 4 Bytes, as a
synchronization mechanism. Also, see *noc_semaphore_set*.

## `ttkernel.experimental.setup_fabric_connections`

SetupFabricConnections

Setup fabric connections for inter-device communication. The connection scheme
is selected by the owning kernel's `fabric_config_index`, which indexes into
`EnqueueProgramOp`'s `fabricConnectionConfigs` table.

`FabricConnectionConfig` specifies:
  - `noc_index`: Which NOC the fabric uses (must match kernel's NocConfig)
  - `topology`: The routing scheme to use for the mesh device (e.g. Line, Ring)
  - `cluster_axis`: The axis along which the to route for 1D topologies
  - `num_links`: Number of routing planes (connections to fabric routers)

After setup, the `FabricConnectionManager` can be used with ops like
`fabric_fast_write_any_len` to send data to other devices in the mesh.

## `ttkernel.experimental.tilize_block`

Experimental TilizeBlockOp call.

This is a custom tilize block LLK that takes the dimensions of the block, and properly tilizes each row.

## `ttkernel.experimental.untilize_block`

Experimental UntilizeBlockOp call.

This is a custom untilize block LLK that takes the dimensions of the block.

## `ttkernel.experimental.write_col_mask_tile`

Experimental Write Col Mask Tile Op

Writes a column mask tile pattern to a CB, where element[i,j] = 1.0 if j < validCols, else 0.0.
This is used in dataflow kernels to generate OOB mask tiles directly in L1 memory,
avoiding DST register pressure in the compute kernel.

## `ttkernel.experimental.write_row_mask_tile`

Experimental Write Row Mask Tile Op

Writes a row mask tile pattern to a CB, where element[i,j] = 1.0 if i < validRows, else 0.0.
This is used in dataflow kernels to generate OOB mask tiles directly in L1 memory,
avoiding DST register pressure in the compute kernel.

## `ttkernel.expm1_tile`

Expm1 operation

Performs element-wise computation of exp(x)-1 on each element of a tile
in DST register at index tile_index. The DST register buffer must be in
acquired state via *tile_regs_acquire* call. This call is blocking and is
only available on the compute engine.

## `ttkernel.expm1_tile_init`

Short init function which configures compute unit for execution of expm1_tile.

Must be run before expm1_tile.

## `ttkernel.fill_tile`

Fill tile with specified value.

Fills supplied DST register tile with a supplied f32 value. The DST
register must be in acquired state via *tile_regs_acquire* call.

Example:
```
ttkernel.fill_tile(%dst_index, %value);
```

## `ttkernel.fill_tile_init`

Init function for fill_tile operation. Refer to documentation for any init function.

Must be run before fill_tile.

## `ttkernel.fill_tile_int`

Fill tile with specified int32 value.

Fills supplied DST register tile with a supplied i32 value. The DST
register must be in acquired state via *tile_regs_acquire* call.

## `ttkernel.float32_greater`

Compare two float32 values using integer arithmetic.

Returns true if f32_a > f32_b.  Operates on raw bits in int32 format.  Maps to float32_greater() in device code.

## `ttkernel.floor_tile`

Floor tile in the DST at specified index.

Performs element-wise computation of floor operation
DST[dst0_index] <- floor(DST[dst0_index])
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.

## `ttkernel.frac_tile`

Frac operation

Performs element-wise computation of the fractional part (x - trunc(x))
on each element of a tile in DST register at index tile_index.
Requires rounding_op_tile_init to be called first. The DST register
buffer must be in acquired state via *tile_regs_acquire* call. This call
is blocking and is only available on the compute engine.

## `ttkernel.ge_binary_tile`

Greater-than-or-equal operation between two tiles

Performs element-wise computation of greater-than-or-equal operation
DST[odst_index] <- DST[dst0_index] >= DST[dst1_index]
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.

## `ttkernel.ge_binary_tile_init`

Short init function

Must be run before ge_binary_tile.

## `ttkernel.gelu_tile`

GELU operation

Performs element-wise computation of GELU on each element of a tile
in DST register at index tile_index. The DST register buffer must be in
acquired state via *tile_regs_acquire* call. This call is blocking and is only
available on the compute engine.

## `ttkernel.gelu_tile_init`

Short init function which configures compute unit for execution of gelu_tile.

Must be run before gelu_tile.

## `ttkernel.get_arg_val`

Get runtime arg value.

Get runtime argument value at specified index.

## `ttkernel.get_common_arg_val`

Get common runtime arg value.

Get runtime argument value at specified index. (Indexes from different location compared to get_arg_val)

## `ttkernel.get_compile_time_arg_val`

Get compile-time arg value.

Get compile-time argument value at specified index.

## `ttkernel.get_dataformat`

Get the data format of a given CB

get_dataformat operation

## `ttkernel.get_noc_addr`

GetNocAddr

GetNocAddr api including core coordinates.

## `ttkernel.get_noc_addr_from_bank_id`

GetNocAddrFromBankID

GetNocAddrFromBankID api

## `ttkernel.get_noc_multicast_addr`

GetNocMulticastAddr

Default tt-metal get_noc_multicast_addr.

The caller must make sure the start and end coordinates are flipped on
WH/BH's NoC1, especially when the optional `noc` operand is absent.

## `ttkernel.get_read_ptr`

GetReadPtr

GetReadPtr operation

## `ttkernel.get_semaphore`

GetSemaphoreOp

Get L1 addr of the semaphore with specified semaphore id

## `ttkernel.get_tile_size`

Get the tile size in bytes of a given CB

get_tile_size operation

## `ttkernel.get_write_ptr`

GetWritePtr

GetWritePtr operation

## `ttkernel.gez_tile`

Greater than or equal to zero tile in the DST at specified index.

Performs element-wise greater than or equal to zero comparison on DST register tiles.
DST[dst0_index] <- (DST[dst0_index] >= 0)
The DST register buffer must be in acquired state via *tile_regs_acquire* call.
This call is blocking and is only available on the compute engine.

## `ttkernel.gez_tile_init`

Init function for gez() operation. Refer to documentation for any init function.

Must be run before gez_tile.

## `ttkernel.gez_tile_int32`

Greater than or equal to zero tile in the DST at specified index.

Performs element-wise greater than or equal to zero comparison on DST register tiles for int32 data type.
DST[dst0_index] <- (DST[dst0_index] >= 0)
The DST register buffer must be in acquired state via *tile_regs_acquire* call.
This call is blocking and is only available on the compute engine.

## `ttkernel.gt_binary_tile`

Greater-than operation between two tiles

Performs element-wise computation of greater-than operation
DST[odst_index] <- DST[dst0_index] > DST[dst1_index]
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.

## `ttkernel.gt_binary_tile_init`

Short init function

Must be run before gt_binary_tile.

## `ttkernel.gtz_tile`

Greater than zero tile in the DST at specified index.

Performs element-wise greater than zero comparison on DST register tiles.
DST[dst0_index] <- (DST[dst0_index] > 0)
The DST register buffer must be in acquired state via *tile_regs_acquire* call.
This call is blocking and is only available on the compute engine.

## `ttkernel.gtz_tile_init`

Init function for gtz() operation. Refer to documentation for any init function.

Must be run before gtz_tile.

## `ttkernel.gtz_tile_int32`

Greater than zero tile in the DST at specified index.

Performs element-wise greater than zero comparison on DST register tiles for int32 data type.
DST[dst0_index] <- (DST[dst0_index] > 0)
The DST register buffer must be in acquired state via *tile_regs_acquire* call.
This call is blocking and is only available on the compute engine.

## `ttkernel.hardsigmoid_tile`

Hardsigmoid operation

Performs element-wise computation of hardsigmoid on each element of a tile
in DST register at index tile_index. The DST register buffer must be in
acquired state via *tile_regs_acquire* call. This call is blocking and is only
available on the compute engine.

## `ttkernel.hardsigmoid_tile_init`

Short init function which configures compute unit for execution of hardsigmoid_tile.

Must be run before hardsigmoid_tile.

## `ttkernel.init_sfpu`

Initialization function for SFPU operations.

This operation initializes all necessary components for SFPU operations,
including unpacking, packing, and math configurations.

## `ttkernel.invoke_sfpi`

let arguments = (ins);

let regions = (region AnyRegion:$region);

let assemblyFormat = [{
  attr-dict-with-keyword $region

## `ttkernel.le_binary_tile`

Less-than-or-equal operation between two tiles

Performs element-wise computation of less-than-or-equal operation
DST[odst_index] <- DST[dst0_index] <= DST[dst1_index]
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.

## `ttkernel.le_binary_tile_init`

Short init function

Must be run before le_binary_tile.

## `ttkernel.lez_tile`

Less than or equal to zero tile in the DST at specified index.

Performs element-wise less than or equal to zero comparison on DST register tiles.
DST[dst0_index] <- (DST[dst0_index] <= 0)
The DST register buffer must be in acquired state via *tile_regs_acquire* call.
This call is blocking and is only available on the compute engine.

## `ttkernel.lez_tile_init`

Init function for lez() operation. Refer to documentation for any init function.

Must be run before lez_tile.

## `ttkernel.lez_tile_int32`

Less than or equal to zero tile in the DST at specified index.

Performs element-wise less than or equal to zero comparison on DST register tiles for int32 data type.
DST[dst0_index] <- (DST[dst0_index] <= 0)
The DST register buffer must be in acquired state via *tile_regs_acquire* call.
This call is blocking and is only available on the compute engine.

## `ttkernel.load_from_l1`

LoadFromL1

Load value from L1.

## `ttkernel.log1p_tile`

Log1p operation

Performs element-wise computation of log(1+x) on each element of a tile
in DST register at index tile_index. The DST register buffer must be in
acquired state via *tile_regs_acquire* call. This call is blocking and is
only available on the compute engine.

## `ttkernel.log1p_tile_init`

Short init function which configures compute unit for execution of log1p_tile.

Must be run before log1p_tile.

## `ttkernel.log_tile`

Log operation

Performs element-wise computation of log on each element of a tile
in DST register at index tile_index. The DST register buffer must be in
acquired state via *tile_regs_acquire* call. This call is blocking and is only
available on the compute engine.

## `ttkernel.log_tile_init`

Short init function which configures compute unit for execution of log_tile.

Must be run before log_tile.

## `ttkernel.logical_not_tile`

Logical negation tile in the DST at specified index.

Performs element-wise computation of logical negation operation
DST[dst0_index] <- !DST[dst0_index]
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.
The DataFormat template parameter specifies the data type.

## `ttkernel.logical_not_tile_init`

Init function for logical_not_tile operation. Refer to documentation for any init function.

Must be run before logical_not_tile.

## `ttkernel.lt_binary_tile`

Less-than operation between two tiles

Performs element-wise computation of less-than operation
DST[odst_index] <- DST[dst0_index] < DST[dst1_index]
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.

## `ttkernel.lt_binary_tile_init`

Short init function

Must be run before lt_binary_tile.

## `ttkernel.ltz_tile`

Less than zero tile in the DST at specified index.

Performs element-wise less than zero comparison on DST register tiles.
DST[dst0_index] <- (DST[dst0_index] < 0)
The DST register buffer must be in acquired state via *tile_regs_acquire* call.
This call is blocking and is only available on the compute engine.

## `ttkernel.ltz_tile_init`

Init function for ltz() operation. Refer to documentation for any init function.

Must be run before ltz_tile.

## `ttkernel.ltz_tile_int32`

Less than zero tile in the DST at specified index.

Performs element-wise less than zero comparison on DST register tiles for int32 data type.
DST[dst0_index] <- (DST[dst0_index] < 0)
The DST register buffer must be in acquired state via *tile_regs_acquire* call.
This call is blocking and is only available on the compute engine.

## `ttkernel.matmul_block`

Block matmul operation

Performs block-sized matrix multiplication *C=A\*B* between blocks of tiles
in two input CBs and accumulates the result into DST. Block dimensions are
specified by ct_dim (output columns in tiles), rt_dim (output rows in tiles),
and kt_dim (inner dimension in tiles). The transpose flag controls whether
input 1 is transposed. The DST register buffer must be in acquired state via
ttkernel.tile_regs_acquire. Blocking; compute engine only.

## `ttkernel.matmul_tiles`

Matmul tiles operation

Performs tile-sized matrix multiplication *C=A\*B* between the tiles in two
specified input CBs and writes the result to DST. The DST register buffer
must be in acquired state via ttkernel.tile_regs_acquire call. This call is blocking and
is only available on the compute engine.

## `ttkernel.max_reduce_with_indices`

Max reduce with indices operation

Performs MaxPool with indices algorithm on the data tile and index tile
that are pre-loaded in DST register. The DST register buffer must be in
acquired state via *acquire_dst* call. This call is blocking and is only
available on the compute engine. The chunk operand is only used when the
accumulate flag is true, controlling when the intermediate data is stored
to the accumulators. When accumulate is true, the chunk operand must be
zero on the first iteration of a reduction loop and non-zero for the rest
of the loop.

## `ttkernel.max_reduce_with_indices_init`

Init for max_reduce_with_indices.

Initializes max_reduce_with_indices. Must be called before max_reduce_with_indices.

## `ttkernel.mem_zeros_base`

Op corresponding to MEM_ZEROS_BASE macro in kernels.

Op corresponding to MEM_ZEROS_BASE macro in kernels.

## `ttkernel.mem_zeros_size`

Op corresponding to MEM_ZEROS_SIZE macro in kernels.

Op corresponding to MEM_ZEROS_SIZE macro in kernels.

## `ttkernel.mm_block_init`

Matmul init function

Initialization for matmul_block operation. Must be called before matmul_block.

## `ttkernel.mm_block_init_short`

Matmul short init function

A short version of matmul_block initialization.

## `ttkernel.mm_init`

Matmul init function

Can only be run ONCE per kernel. Should be run before matmul.

## `ttkernel.mm_init_short`

Matmul short init function

Can be run MULTIPLE times per kernel. Should be run before matmul. Use this if some other init was called between mm_init and matmul_tiles. (i.e. in a loop)

## `ttkernel.mul_binary_tile`

Multiplication operation between two tiles

Performs element-wise computation of multiplication operation
DST[odst_index] <- DST[dst0_index] * DST[dst1_index]
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.

## `ttkernel.mul_binary_tile_init`

Short init function

Must be run before mul_binary_tile.

## `ttkernel.mul_int_tile`

Integer multiplication operation between two tiles

Performs element-wise computation of integer multiplication operation
DST[odst_index] <- DST[dst0_index] * DST[dst1_index]
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.
Supported data formats are: Int32, UInt32, UInt16.

## `ttkernel.mul_int_tile_init`

Short init function

Must be run before mul_int_tile.

## `ttkernel.mul_tiles`

Mul operation

Performs element-wise multiplication C=A*B of tiles in two CBs at given
indices and writes the result to the DST register at index dst_tile_index.
The DST register buffer must be in acquired state via *tile_regs_acquire* call.
This call is blocking and is only available on the compute engine.

## `ttkernel.mul_tiles_init`

Short init function

Must be run before mul_tiles.

## `ttkernel.mul_unary_tile`

Multiply by scalar operation

Performs element-wise multiplication of a tile by a scalar value.
DST[dst0_index] <- DST[dst0_index] * scalar
The DST register buffer must be in acquired state via *tile_regs_acquire* call.
This call is blocking and is only available on the compute engine.

## `ttkernel.my_logical_x_`

MyLogicalX

Lowers to the tt-metal supported my_logical_x_ global. This represents the logical X coordinate of the current core.

## `ttkernel.my_logical_y_`

MyLogicalY

Lowers to the tt-metal supported my_logical_y_ global. This represents the logical Y coordinate of the current core.

## `ttkernel.my_x`

MyX

Lowers to the tt-metal supported MY_X macro. This represents the virtual X coordinate of the current core.

## `ttkernel.my_y`

MyY

Lowers to the tt-metal supported MY_Y macro. This represents the virtual Y coordinate of the current core.

## `ttkernel.ne_binary_tile`

Inequality operation between two tiles

Performs element-wise computation of inequality operation
DST[odst_index] <- DST[dst0_index] != DST[dst1_index]
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.

## `ttkernel.ne_binary_tile_init`

Short init function

Must be run before ne_binary_tile.

## `ttkernel.negative_tile`

Negative operation

Performs element-wise computation of the negative on each element of a tile
in DST register at index tile_index. The DST register buffer must be in
acquired state via *acquire_dst* call. This call is blocking and is only
available on the compute engine.

## `ttkernel.negative_tile_init`

Short init function which configures compute unit for execution of negative_tile.

Must be run before negative_tile.

## `ttkernel.negative_tile_int32`

Negative i32 tile in the DST at specified index.

Performs element-wise computation of negation operation
DST[dst0_index] <- -DST[dst0_index]
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.

## `ttkernel.nez_tile`

Not equal to zero tile in the DST at specified index.

Performs element-wise inequality on DST register tiles.
DST[dst0_index] <- (DST[dst0_index] != 0)
The DST register buffer must be in acquired state via *tile_regs_acquire* call.
This call is blocking and is only available on the compute engine.

## `ttkernel.nez_tile_init`

Init function for nez() operation. Refer to documentation for any init function.

Must be run before nez_tile.

## `ttkernel.nez_tile_int32`

Not equal to zero tile in the DST at specified index.

Performs element-wise inequality on DST register tiles for int32 data type.
DST[dst0_index] <- (DST[dst0_index] != 0)
The DST register buffer must be in acquired state via *tile_regs_acquire* call.
This call is blocking and is only available on the compute engine.

## `ttkernel.noc_async_atomic_barrier`

NocAsyncAtomicBarrier

Block until all outstanding non-posted atomic operations on the given
NOC have flushed. Pairs with `noc_semaphore_inc` and
`noc_semaphore_inc_multicast`.

## `ttkernel.noc_async_read`

NocAsyncRead

NocAsyncRead from either an L1 core coordinate or a DRAM bank.

## `ttkernel.noc_async_read_barrier`

NocAsyncReadBarrier

Waits for all outstanding read transactions on the given NOC.

## `ttkernel.noc_async_read_barrier_with_trid`

NocAsyncReadBarrierWithTrid

Waits for outstanding read transactions matching a transaction ID. TRID is 0-15, NOC is 0 or 1.

Example:
```
ttkernel.noc_async_read_barrier_with_trid(%trid, %noc_idx) : (i32, i8) -> ()
```

## `ttkernel.noc_async_read_one_packet_set_state`

NocAsyncReadOnePacketSetState

NocAsyncReadOnePacketSetState

## `ttkernel.noc_async_read_one_packet_with_state`

NocAsyncReadOnePacketWithState

NocAsyncReadOnePacketWithState

## `ttkernel.noc_async_read_tile`

NocAsyncReadTile

NocAsyncReadTile

## `ttkernel.noc_async_write`

NocAsyncWrite

NocAsyncWrite to either an L1 core coordinate or a DRAM bank.

## `ttkernel.noc_async_write_barrier`

NocAsyncWriteBarrier

Waits for all outstanding write transactions on the given NOC.

## `ttkernel.noc_async_write_barrier_with_trid`

NocAsyncWriteBarrierWithTrid

Waits for outstanding write transactions matching a transaction ID. TRID is 0-15, NOC is 0 or 1.

Example:
```
ttkernel.noc_async_write_barrier_with_trid(%trid, %noc_idx) : (i32, i8) -> ()
```

## `ttkernel.noc_async_write_multicast`

NocAsyncWriteMulticast

Initiates an asynchronous write from a source address in L1 memory on the
Tensix core executing this function call to a rectangular destination grid.
The destinations are specified using an on-chip grid of nodes located at
NOC coordinate range (x_start,y_start,x_end,y_end) and a local address.
Also, *see noc_async_write_barrier*.

The caller must make sure the start and end coordinates are flipped on
WH/BH's NoC1, especially when the optional `noc` operand is absent.

The destination nodes can only be a set of Tensix cores + L1 memory address.
The destination nodes must form a rectangular grid. The destination L1
memory address must be the same on all destination nodes.

With this API, the multicast sender cannot be part of the multicast
destinations. If the multicast sender has to be in the multicast
destinations (i.e. must perform a local L1 write), the other API variant
*noc_async_write_multicast_loopback_src* can be used.

Note: The number of destinations needs to be non-zero. Besides that,
there is no restriction on the number of destinations, i.e. the
multicast destinations can span the full chip. However, as mentioned
previously, the multicast source cannot be part of the destinations. So, the
maximum number of destinations is 119.

## `ttkernel.noc_async_write_multicast_loopback_src`

NocAsyncWriteMulticastLoopbackSrc

Multicast write that allows the sender to be part of the destinations.

The caller must make sure the start and end coordinates are flipped on
WH/BH's NoC1, especially when the optional `noc` operand is absent.

## `ttkernel.noc_async_write_multicast_one_packet`

NocAsyncWriteMulticastOnePacket

NocAsyncWriteMulticastOnePacket
this issues only a single packet with size <= NOC_MAX_BURST_SIZE (ie maximum packet size)

## `ttkernel.noc_async_write_one_packet_with_trid`

NocAsyncWriteOnePacketWithTrid

Issues a one-packet NOC write with a specific transaction ID. TRID is 0-15, NOC is 0 or 1.

Example:
```
ttkernel.noc_async_write_one_packet_with_trid(%l1_src, core[%x, %y], %dst_addr, %size, %trid, noc %noc_idx) : (i32, index, index, i32, i32, i32, i8) -> ()
// TRID-specific barrier should follow.
// ttkernel.noc_async_write_barrier_with_trid(%trid, %noc_idx) : (i32, i8) -> ()
```

## `ttkernel.noc_async_write_tile`

NocAsyncWriteTile

NocAsyncWriteTilie

## `ttkernel.noc_inline_dw_write`

NocInlineDwWrite

Initiates an inline 32-bit NOC write to a remote L1 address. The value is
provided as an SSA value, so the operation does not require a local SRAM
staging word.

## `ttkernel.noc_semaphore_inc`

NocSemaphoreInc

The Tensix core executing this function call initiates an atomic increment
(with 32-bit wrap) of a remote Tensix core L1 memory address. This L1 memory
address is used as a semaphore of size 4 Bytes, as a synchronization
mechanism.

The `posted` attribute selects the NOC transaction response mode and
maps to the homonymous template parameter of `noc_semaphore_inc` in
tt-metal. When unset or false, the transaction is non-posted: the
receiver returns an acknowledgement and the sender's outstanding-atomics
counter tracks completion. When true, the transaction is posted: no
acknowledgement is returned, the sender's counter is not updated, and
synchronization must be receiver-driven.

## `ttkernel.noc_semaphore_inc_multicast`

NocSemaphoreIncMulticast

Initiates an atomic increment (with 32-bit wrap) of a 4-byte L1 semaphore
on every core in a rectangular destination grid. The destinations are
encoded in a uint64_t produced by *get_noc_multicast_addr* covering the
NOC coordinate range (x_start, y_start, x_end, y_end) and the target L1
address. Used for cumulative synchronization across multiple senders
sharing receivers, paired with *experimental.semaphore_wait_min*.

The sender must not be a member of the multicast destination set; for
including the sender there is no corresponding loopback variant in
tt-metal at this time.

The `posted` attribute mirrors the semantics of `noc_semaphore_inc`'s
`posted` template parameter. Posted multicast is currently restricted
on some architectures in the underlying tt-metal (cf. the assertion in
the v2 `Noc::async_write_multicast` API); leave unset when in doubt.

## `ttkernel.noc_semaphore_set`

NocSemaphoreSet

Sets the value of a local L1 memory address on the Tensix core executing
this function to a specific value. This L1 memory address is used as a
semaphore of size 4 Bytes, as a synchronization mechanism. Also, see
*noc_semaphore_wait*.

## `ttkernel.noc_semaphore_set_multicast`

NocSemaphoreSetMulticast

Initiates an asynchronous write from a source address in L1 memory on the
Tensix core executing this function call to a rectangular destination grid.
The destinations are specified using a uint64_t encoding referencing an
on-chip grid of nodes located at NOC coordinate range
(x_start,y_start,x_end,y_end) and a local address created using
*get_noc_multicast_addr* function. The size of data that is sent is 4 Bytes.
This is usually used to set a semaphore value at the destination nodes, as a
way of a synchronization mechanism. The same as *noc_async_write_multicast*
with preset size of 4 Bytes.
With this API, the multicast sender cannot be part of the multicast
destinations. If the multicast sender has to be in the multicast
destinations (i.e. must perform a local L1 write), the other API variant
*noc_semaphore_set_multicast_loopback_src* can be used.

## `ttkernel.noc_semaphore_set_multicast_loopback_src`

NocSemaphoreSetMulticastLoopback

Initiates an asynchronous write from a source address in L1 memory on the
Tensix core executing this function call to a rectangular destination grid.
The destinations are specified using a uint64_t encoding referencing an
on-chip grid of nodes located at NOC coordinate range
(x_start,y_start,x_end,y_end) and a local address created using
*get_noc_multicast_addr* function. The size of data that is sent is 4 Bytes.
This is usually used to set a semaphore value at the destination nodes, as a
way of a synchronization mechanism. The same as *noc_async_write_multicast*
with preset size of 4 Bytes.
Note: With this API, sending data only to the source node (when num_dests
is 1) may result in unexpected behaviour. For some parameters, hangs have
been observed. For some other parameters, nothing may happen. Consider using
regular non multicast operations such as *noc_async_write* in this case.

## `ttkernel.pack_reconfig_data_format`

Reconfigure packer data format for the given output CB.

Reconfigures the packer data format to match the output circular buffer.
Must be called before pack_reconfig_l1_acc to ensure format consistency
during L1 accumulation.

## `ttkernel.pack_reconfig_l1_acc`

Reconfigure packer to L1 accumulation mode.

Reconfigures the packer to accumulate when packing from DST to L1.

## `ttkernel.pack_tile`

PackTile op.

Copies a single tile from the DST register buffer at a specified index to a
specified CB at a given index. For the out_tile_index to be valid for this
call, cb_reserve_back(n) has to be called first to reserve at least some
number n > 0 of tiles in the output CB. out_tile_index = 0 then references
the first tile in the reserved section of the CB, up to index n - 1, which will
then be visible to the consumer in the same order after a cb_push_back call.
The DST register buffer must be in acquired state via *tile_regs_acquire* call.
This call is blocking and is only available on the compute engine.

Each subsequent pack call will increment the write pointer in the cb by single
tile size. The pointer is then again set to a valid position with space for n
reserved tiles by another cb_reserve_back call.

Operates in tandem with functions cb_reserve_back and cb_push_back.

A typical use case is first the producer ensures that there is a number of
tiles available in the buffer via cb_reserve_back, then the producer uses
the pack_tile call to copy a tile from one of DST slots to a slot in
reserved space and finally cb_push_back is called to announce visibility of
the reserved section of the circular buffer to the consumer.

## `ttkernel.pack_tile_block`

PackTileBlock op.

Copies a contiguous block of ntiles tiles from the DST register buffer,
starting at dst_index through dst_index + ntiles - 1, to a specified
output CB. Unlike pack_tile, which copies a single tile and supports
out-of-order packing, this copies tiles sequentially from contiguous DST
slots. The CB write pointer advances by ntiles after each call and resets
after cb_push_back.

The DST register buffer must be in acquired state via *tile_regs_acquire* call.
This call is blocking and is only available on the compute engine.

Operates in tandem with functions cb_reserve_back and cb_push_back.

A typical use case is packing an entire block of computed tiles from DST
to an output CB in one call, rather than issuing pack_tile for each tile
individually. The producer first reserves space via cb_reserve_back, then
calls pack_tile_block to copy all ntiles at once, and finally calls
cb_push_back to make the block visible to the consumer.

## `ttkernel.pack_untilize_init`

PackUntilizeInitOp call.

Initializes UNPACK, MATH, and PACK threads for the pack untilize
operation.  Maps to the official tt-metal
`pack_untilize_init<block_ct_dim, full_ct_dim>(icb, ocb)` API.
`cols_per_dst_pass` is the number of column tiles processed per DST pass
(`block_ct_dim` in the tt-metal template), and `total_col_tiles` is the
total number of column tiles (`full_ct_dim` in the tt-metal template).
For correctness, `cols_per_dst_pass` must divide `total_col_tiles`, and
`cols_per_dst_pass` must fit
in DST capacity for the target data type. Both default to 1.
This op is expected to be paired with
`experimental.pack_untilize_block` and finalized by
`pack_untilize_uninit`.

## `ttkernel.pack_untilize_uninit`

PackUntilizeUninitOp call.

Uninitializes the pack untilize operation.  Maps to the official tt-metal
`pack_untilize_uninit(ocb)` API.

## `ttkernel.power_binary_tile`

Power operation between two tiles

Performs element-wise computation of power operation
DST[odst_index] <- DST[dst0_index] ^ DST[dst1_index]
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.

## `ttkernel.power_binary_tile_init`

Short init function which configures compute unit for execution of power_binary_tile.

Must be run before power_binary_tile.

## `ttkernel.power_tile`

Power by scalar operation

Performs element-wise exponentiation of a tile by a scalar value.
DST[dst0_index] <- DST[dst0_index] ^ scalar
The DST register buffer must be in acquired state via *tile_regs_acquire* call.
This call is blocking and is only available on the compute engine.

## `ttkernel.power_tile_init`

Init function for power_tile operation.

Must be run before power_tile.

## `ttkernel.rand_tile`

Rand tile operation

Performs element-wise rand on each element of a tile in DST register at
index tile_index. That is each element is overwritten with a randomly
generated float in the range [from, from + scale).

`from` and `scale` are passed as the uint32_t bit pattern of their
corresponding f32 values. The DST register buffer must be in acquired
state via *tile_regs_acquire* call.

## `ttkernel.rand_tile_init`

Init function for rand_tile operation. Refer to documentation for any init function.

Initializes the PRNG seed for the SFPU. Must be run before rand_tile.

## `ttkernel.recip_tile`

Recip tile in the DST at specified index.

Performs element-wise computation of the reciprocal on each element of a tile
in DST register at index tile_index. The DST register buffer must be in
acquired state via *tile_regs_acquire* call. This call is blocking and is only
available on the compute engine.
Only works for Float32, Float16_b, Bfp8_b data formats for full accuracy.

## `ttkernel.recip_tile_init`

Init function for recip_tile operation. Refer to documentation for any init function.

Must be called before recip_tile function.

## `ttkernel.reconfig_data_format_srca`

Reconfigure Src A data format for the given input CB.

Reconfigures the srcA data format for a new operand, always re-deriving the int8/unsigned state from the new format.

## `ttkernel.reduce_init`

Init function

Must be run before reduce_tile.

## `ttkernel.reduce_tile`

Reduce operation

Performs a reduction operation *B = reduce(A)* using reduce_func for
dimension reduction on a tile in the CB at a given index and writes the
result to the DST register at index *dst_tile_index*. Reduction can be
either of type *Reduce::R*, *Reduce::C* or *Reduce::RC*, identifying the
dimension(s) to be reduced in size to 1. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.
The templates takes reduce_type which can be ReduceFunc::Sum, ReduceFunc::Max
and reduce_dim which can be Reduce::R, Reduce::C, Reduce::RC.
They can also be specified by defines REDUCE_OP and REDUCE_DIM.
This call is blocking and is only available on the compute engine.

## `ttkernel.reduce_uninit`

Init function for reduce_uninit operation.

Resets the packer edge mask configuration to its default state by clearing any previously set masks. Needs to be called after
reduce_tile if the next operation requires default packer state. In case that the next operation is reduce operation across the
same dimension, this call can be omitted. If this function is not called, the packer will continue to use the edge masks set
by the latest reduce_init call, which may lead to incorrect packing behavior in subsequent operations.

This function is not in line with our programming mode. To be removed by end of 2025. tt-metal#22904.

## `ttkernel.reinterpret_cast`

CastToL1Ptr

Cast specified addr to L1 pointer.

## `ttkernel.relu_tile`

Relu operation

Performs element-wise computation of relu on each element of a tile
in DST register at index tile_index. The DST register buffer must be in
acquired state via *tile_regs_acquire* call. This call is blocking and is only
available on the compute engine.

## `ttkernel.relu_tile_init`

Short init function which configures compute unit for execution of relu_tile(_int32).

Must be run before relu_tile(_int32).

## `ttkernel.relu_tile_int32`

Relu operation (for int32 type)

Performs element-wise computation of relu on each element of a tile
in DST register at index tile_index. The DST register buffer must be in
acquired state via *tile_regs_acquire* call. This call is blocking and is only
available on the compute engine.

## `ttkernel.remote_sram_write_u32`

RemoteSramWriteU32

Initiates a 4-byte remote SRAM write of the u32 value stored at a local
SRAM address on the Tensix core executing this function call. The source
SRAM word may be referenced by a typed SRAM address, a computed i32 SRAM
address, or a local semaphore handle.

## `ttkernel.rounding_op_tile_init`

Init function for ceil/floor/round_tile operation. Refer to documentation for any init function.

Must be run before ceil/floor/round_tile.

## `ttkernel.rsqrt_tile`

Rsqrt operation

Performs element-wise computation of reciprocal sqrt on each element of a tile
in DST register at index tile_index. The DST register buffer must be in
acquired state via *acquire_dst* call. This call is blocking and is only
available on the compute engine.

## `ttkernel.rsqrt_tile_init`

Short init function which configures compute unit for execution of rsqrt_tile.

Must be run before rsqrt_tile.

## `ttkernel.selu_tile`

Selu operation

Performs element-wise SELU activation: scale*(max(0,x) + min(0,alpha*(exp(x)-1)))
on each element of a tile in DST register at index tile_index.
scale and alpha are uint32 bit-cast representations of bf16 float values.
The DST register buffer must be in acquired state via *tile_regs_acquire*
call. This call is blocking and is only available on the compute engine.

## `ttkernel.selu_tile_init`

Short init function which configures compute unit for execution of selu_tile.

Must be run before selu_tile.

## `ttkernel.sfpu_reduce`

SFPU reduce operation on a tile in DST.

Performs an intra-tile reduction on the tile at index `dst_index` in the
DST register using the SFPU path. The input tile must already be loaded
into DST (e.g. via copy_tile). The reduction is performed in-place on
that DST slot, writing the result into the first row (REDUCE_COL) or
first column (REDUCE_ROW) of the same tile.

Only 32x32 tile dimensions are supported.

Only Sum/Max reductions over Int32 tiles are supported today, and
reduce_dim must be Row or Col (Scalar must be decomposed into Col + Row
by the caller); the verifier enforces these constraints.

Lowers to: sfpu_reduce<PoolType, DataFormat, ReduceDim>(dst_index).

## `ttkernel.sfpu_reduce_init`

Init for sfpu_reduce.

Initializes the SFPU reduce kernel. Must be called before sfpu_reduce.
Unlike reduce_init this call carries no CB/DST operands: it only
configures the SFPU for the given PoolType and DataFormat.

Only Sum/Max reductions over Int32 tiles are supported today; the
verifier enforces this.

Lowers to: sfpu_reduce_init<PoolType, DataFormat>().

## `ttkernel.sigmoid_tile`

Sigmoid operation

Performs element-wise computation of sigmoid on each element of a tile
in DST register at index tile_index. The DST register buffer must be in
acquired state via *tile_regs_acquire* call. This call is blocking and is only
available on the compute engine.

## `ttkernel.sigmoid_tile_init`

Short init function which configures compute unit for execution of sigmoid_tile.

Must be run before sigmoid_tile.

## `ttkernel.sign_tile`

Sign operation

Performs element-wise computation of sign on each element of a tile
in DST register at index tile_index. The DST register buffer must be in
acquired state via *tile_regs_acquire* call. This call is blocking and is only
available on the compute engine.

## `ttkernel.sign_tile_init`

Init function for sign_tile operation. Refer to documentation for any init function.

Must be run before sign_tile.

## `ttkernel.signbit_tile`

Signbit operation

Performs element-wise sign bit extraction on each element of a tile
in DST register at index tile_index, returning 0.0 if the sign bit is
clear and 1.0 if it is set (matching IEEE 754 signbit semantics).
The DST register buffer must be in acquired state via *tile_regs_acquire*
call. This call is blocking and is only available on the compute engine.

## `ttkernel.signbit_tile_init`

Short init function which configures compute unit for execution of signbit_tile.

Must be run before signbit_tile.

## `ttkernel.silu_tile`

Silu operation

Performs element-wise computation of silu on each element of a tile
in DST register at index tile_index. The DST register buffer must be in
acquired state via *tile_regs_acquire* call. This call is blocking and is only
available on the compute engine.

## `ttkernel.silu_tile_init`

Short init function which configures compute unit for execution of silu_tile.

Must be run before silu_tile.

## `ttkernel.sin_tile`

Sine tile in the DST at specified index.

Performs element-wise computation of sine operation
DST[dst0_index] <- sin(DST[dst0_index])
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.

## `ttkernel.sin_tile_init`

Init function for sin_tile operation. Refer to documentation for any init function.

Must be run before sin_tile.

## `ttkernel.softsign_tile`

Softsign operation

Performs element-wise computation of x/(1+|x|) on each element of a tile
in DST register at index tile_index. The DST register buffer must be in
acquired state via *tile_regs_acquire* call. This call is blocking and is
only available on the compute engine.

## `ttkernel.softsign_tile_init`

Short init function which configures compute unit for execution of softsign_tile.

Must be run before softsign_tile.

## `ttkernel.sqrt_tile`

Sqrt operation

Performs element-wise computation of sqrt on each element of a tile
in DST register at index tile_index. The DST register buffer must be in
acquired state via *acquire_dst* call. This call is blocking and is only
available on the compute engine.

## `ttkernel.sqrt_tile_init`

Short init function which configures compute unit for execution of sqrt_tile.

Must be run before sqrt_tile.

## `ttkernel.square_tile`

Square operation

Performs element-wise computation of x^2 on each element of a tile in
DST register at index tile_index. The DST register buffer must be in
acquired state via *tile_regs_acquire* call. This call is blocking and
is only available on the compute engine.

## `ttkernel.square_tile_init`

Short init function which configures compute unit for execution of square_tile.

Must be run before square_tile.

## `ttkernel.store_to_l1`

StoreToL1

Store value to L1.

## `ttkernel.sub_binary_tile`

Subtraction operation between two tiles

Performs element-wise computation of subtraction operation
DST[odst_index] <- DST[dst0_index] - DST[dst1_index]
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.

## `ttkernel.sub_binary_tile_init`

Short init function

Must be run before sub_binary_tile.

## `ttkernel.sub_int_tile`

Integer subtraction operation between two tiles

Performs element-wise computation of integer subtraction operation
DST[odst_index] <- DST[dst0_index] - DST[dst1_index]
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.
Supported data formats are: Int32, UInt32, UInt16.

## `ttkernel.sub_int_tile_init`

Short init function

Must be run before sub_int_tile.

## `ttkernel.sub_tiles`

Sub operation

Performs element-wise subtraction C=A-B of tiles in two CBs at given
indices and writes the result to the DST register at index dst_tile_index.
The DST register buffer must be in acquired state via *tile_regs_acquire* call.
This call is blocking and is only available on the compute engine.

## `ttkernel.sub_tiles_init`

Short init function

Must be run before sub_tiles.

## `ttkernel.sub_unary_tile`

Subtract by scalar operation

Performs element-wise subtraction of a tile by a scalar value.
DST[dst0_index] <- DST[dst0_index] - scalar
The DST register buffer must be in acquired state via *tile_regs_acquire* call.
This call is blocking and is only available on the compute engine.

## `ttkernel.sub_unary_tile_int32`

Subtract by int32 scalar operation

Performs element-wise subtraction of an int32 tile by an int32 scalar value.
DST[dst0_index] <- DST[dst0_index] - scalar
The DST register buffer must be in acquired state via *tile_regs_acquire* call.
This call is blocking and is only available on the compute engine.

## `ttkernel.tan_tile`

Tan operation

Performs element-wise computation of the trigonometric tangent operation on each element of a tile
in DST register at index tile_index. The DST register buffer must be in
acquired state via *acquire_dst* call. This call is blocking and is only
available on the compute engine.

## `ttkernel.tan_tile_init`

Short init function which configures compute unit for execution of tan_tile.

Must be run before tan_tile.

## `ttkernel.tanh_tile`

Tanh operation

Performs element-wise computation of the hyperbolic tangent operation on each element of a tile
in DST register at index tile_index. The DST register buffer must be in
acquired state via *acquire_dst* call. This call is blocking and is only
available on the compute engine.

## `ttkernel.tanh_tile_init`

Short init function which configures compute unit for execution of tanh_tile.

Must be run before tanh_tile.

## `ttkernel.tensor_accessor.get_bank_and_offset`

TensorAccessor's get_bank_and_offset

Returns bank id and page offset.

## `ttkernel.tensor_accessor.get_noc_addr`

TensorAccessor's get_noc_addr

get_noc_addr using information stored in the TensorAccessor.

## `ttkernel.tensor_accessor.get_shard_noc_addr`

TensorAccessor's get_shard_noc_addr

Returns noc addr of a shard.

## `ttkernel.tensor_accessor.is_local_addr`

TensorAccessor's is_local_addr

Returns bool indicating addr locality.

## `ttkernel.tensor_accessor.is_local_bank`

TensorAccessor's is_local_bank

Returns bool indicating bank locality.

## `ttkernel.tensor_accessor.is_local_page`

TensorAccessor's is_local_page

Returns bool indicating page locality.

## `ttkernel.tensor_accessor.is_local_shard`

TensorAccessor's is_local_shard

Returns bool indicating shard locality.

## `ttkernel.tile_regs_acquire`

tile_regs_acquire

Acquire an exclusive lock on the DST register for the MATH thread.
This register is an array of 16 tiles of 32x32 elements each.
This is a blocking function, i.e. this function will wait until the lock is acquired.

## `ttkernel.tile_regs_commit`

tile_regs_commit

Release lock on DST register by MATH thread. The lock had to be previously acquired with tile_regs_acquire.

## `ttkernel.tile_regs_release`

tile_regs_release

Release lock on DST register by PACK thread. The lock had to be previously acquired with tile_regs_wait.

## `ttkernel.tile_regs_wait`

tile_regs_wait

Acquire an exclusive lock on the DST register for the PACK thread.
It waits for the MATH thread to commit the DST register.
This is a blocking function, i.e. this function will wait until the lock is acquired.

## `ttkernel.tilize_block`

TilizeBlockOp call.

TilizeBlockOp operation

## `ttkernel.tilize_init`

TilizeInitOp call.

Initialize the tilize operation. To be called once at beginning of a
kernel.

## `ttkernel.tilize_uninit`

TilizeUninitOp call.

Uninitialize tilize operation before re-initializing for another operation.

## `ttkernel.topk_local_sort`

TopK local sort operation

Bitonic local-sort stage of TopK over the value and index tiles pre-loaded in DST. Compute engine only.

## `ttkernel.topk_merge`

TopK merge operation

Merge stage of TopK combining length-K subsequences in DST. Compute engine only.

## `ttkernel.topk_rebuild`

TopK rebuild operation

Rebuild stage of TopK re-sorting the surviving subsequences in DST. Compute engine only.

## `ttkernel.topk_tile_init`

TopK tile init function

Initialization for the topk operations. Must be called before
topk_local_sort/topk_merge/topk_rebuild.

## `ttkernel.transpose_wh_init`

TransposeInitOp call.

Initialize the transpose operation. To be called once at beginning of a
kernel before transpose operations.

## `ttkernel.transpose_wh_tile`

Transpose WH tile operation

Performs a 32x32 transpose operation on a tile in the CB
at a given index and writes the result to the DST register at index
dst_tile_index.

## `ttkernel.trunc_tile`

Trunc operation

Performs element-wise truncation toward zero on each element of a tile
in DST register at index tile_index.
Requires rounding_op_tile_init to be called first. The DST register
buffer must be in acquired state via *tile_regs_acquire* call. This call
is blocking and is only available on the compute engine.

## `ttkernel.typecast_tile`

Cast the dataformat of the tile in the DST at specified index.

Performs element-wise typecast operation
DST[dst0_index] <- typecast<in_dataformat, out_dataformat>(DST[dst0_index])
on DST register operands. The DST register buffer must be in
acquired state via *tile_regs_acquire* call.

## `ttkernel.typecast_tile_init`

Init function for typecast_tile operation. Refer to documentation for any init function.

Must be run before typecast_tile.

## `ttkernel.unary_bcast`

Broadcast operation

Performs a broadcast operation *B = broadcast(A)* using bcast_dim for
dimension expansion on a tile in the CB at a given index and writes the
result to the DST register at index *dst_tile_index*. The supported
broadcast dimensions are `row`, `col`, `scalar` (both row and column). The
DST register buffer must be in acquired state via *tile_regs_acquire*
call. This call is blocking and is only available on the compute engine.

## `ttkernel.unary_bcast_init`

Init function

Must be run before bcast_tile.

## `ttkernel.unary_op_init_common`

Initialization function for unary operations.

This operation initializes all necessary components for unary operations,
including unpacking, packing, and math configurations.

## `ttkernel.unreachable`

Unreachable op.

Unreachable operation

## `ttkernel.untilize_block`

UntilizeBlockOp call.

UntilizeBlockOp operation

## `ttkernel.untilize_init`

UntilizeInitOp call.

Init function for untilize operations, to be used at the beginning of the
kernel.

## `ttkernel.untilize_uninit`

UntilizeUninitOp call.

Uninitialize untilize operation, to allow initializing another operation.

## `ttkernel.where_tile`

Conditional selection operation

Performs element-wise conditional selection
DST[odst_index] <- condition ? DST[dst_true_index] : DST[dst_false_index]
For each element, if condition is non-zero, selects from dst_true_index,
otherwise selects from dst_false_index.
The DST register buffer must be in acquired state via *acquire_dst* call.

## `ttkernel.where_tile_init`

Short init function

Must be run before where_tile.

