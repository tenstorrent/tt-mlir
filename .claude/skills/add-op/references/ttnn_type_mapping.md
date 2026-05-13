# TTNN C++ to MLIR/Tablegen Type Mapping Reference

This document provides a comprehensive mapping between TTNN C++ library types and their
MLIR/tablegen equivalents in the tt-mlir compiler. Use this as a reference when adding new
ops to understand how C++ parameter types translate to tablegen definitions.

---

## A. Tensor Types

| TTNN C++ Type | MLIR Tablegen Type | Notes |
|---|---|---|
| `const Tensor&` / `Tensor` (required input) | `AnyRankedTensor` | Used for required tensor operands in `ins` |
| `std::optional<const Tensor>` / `std::optional<Tensor>` | `Optional<AnyRankedTensor>` | Used for optional tensor operands (e.g., bias) |
| `Variadic tensors` / `std::vector<Tensor>` | `Variadic<AnyRankedTensor>` | Used for variable-length tensor lists (e.g., ConcatOp inputs) |
| `Tensor` (output) | `AnyRankedTensor` | Used in `outs` for results |
| `Tensor` (DPS output / in-place) | `Arg<AnyRankedTensor, "desc", [MemWrite]>` | For in-place mutation (e.g., cache update ops). The `Arg<>` wrapper adds memory effect annotations. |
| `std::optional<Tensor>` (optional DPS output) | `Arg<Optional<AnyRankedTensor>, "desc", [MemWrite]>` | Optional in-place output tensor |

**Key Points:**
- TTNN C++ functions often have an `optional_output_tensor` DPS parameter for pre-allocated output. This is **not modeled** in most MLIR ops -- the compiler handles output allocation separately via `EmptyOp`.
- In-place ops (UpdateCacheOp, FillCacheOp, WriteTensorOp) use the `TTNN_InplaceOp` base class which adds `MemoryEffects<[MemWrite]>`.

---

## B. Scalar/Attribute Types

| C++ Type | MLIR Tablegen Type | C++ Attr Class | Example Usage |
|---|---|---|---|
| `int32_t` | `I32Attr` | `IntegerAttr` (i32) | `batch_size`, `groups`, `dim0` |
| `int32_t` (signed, explicit) | `SI32Attr` | `IntegerAttr` (si32) | `dimension`, `all_gather_dim` |
| `uint32_t` | `UI32Attr` | `IntegerAttr` (ui32) | `num_heads`, `cluster_axis`, `num_links` |
| `int64_t` | `I64Attr` | `IntegerAttr` (i64) | `start`, `end`, `step` (ArangeOp) |
| `int8_t` (signed) | `SI8Attr` | `IntegerAttr` (si8) | `dim` in SortOp |
| `float` | `F32Attr` | `FloatAttr` (f32) | `epsilon`, `parameter`, `scale`, `value` |
| `bool` | `BoolAttr` | `BoolAttr` | `transpose_a`, `keep_dim`, `is_causal` |
| `std::string` | `StrAttr` | `StringAttr` | `activation`, `mode`, `weights_format` |
| `std::vector<int32_t>` / `SmallVector<int32_t>` | `I32ArrayAttr` | `ArrayAttr` of `IntegerAttr` | `begins`, `ends`, `step` (SliceStaticOp), `shape` (ReshapeOp) |
| `std::vector<int32_t>` (dense) | `DenseI32ArrayAttr` | `DenseI32ArrayAttr` | `kernel_size`, `stride`, `padding`, `dilation` |
| `std::vector<int64_t>` (dense) | `DenseI64ArrayAttr` | `DenseI64ArrayAttr` | `permutation`, `shard_shape`, `shard_dims` |
| `float` or `int32_t` (union) | `AnyAttrOf<[F32Attr, I32Attr]>` | N/A | `rhs` in BinaryCompositeScalarOp, `min`/`max` in ClampScalarOp |
| `int32_t` or `DenseI32ArrayAttr` (union) | `AnyAttrOf<[SI32Attr, DenseI32ArrayAttr]>` | N/A | `scale_factor` in UpsampleOp |
| `mlir::FlatSymbolRefAttr` | `FlatSymbolRefAttr` | `FlatSymbolRefAttr` | `capture_callee`, `execute_callee` |
| `mlir::SymbolRefAttr` | `SymbolRefAttr` | `SymbolRefAttr` | `d2m_func` |

---

## C. TTNN-Specific Types

### C.1 Memory & Layout Types

| TTNN C++ Type | MLIR Tablegen Type | Defined In | Notes |
|---|---|---|---|
| `ttnn::MemoryConfig` | `TTNN_MemoryConfigAttr` | `TTNNOpsAttrs.td` | Composite attr with bufferType, tensorMemoryLayout, optional shardSpec/ndShardSpec |
| `ttnn::BufferType` | `TTNN_BufferTypeAttr` | `TTNNOpsEnums.td` | Enum: DRAM, L1, SystemMemory, L1Small, Trace |
| `ttnn::TensorMemoryLayout` | `TTNN_TensorMemoryLayoutAttr` | `TTNNOpsEnums.td` | Enum: Interleaved, HeightSharded, WidthSharded, BlockSharded |
| `ttnn::Layout` | `TTNN_LayoutAttr` | `TTNNOpsEnums.td` / `TTNNOpsAttrs.td` | Enum: RowMajor, Tile, Invalid |
| `ttnn::ShapeAttr` | `TTNN_ShapeAttr` | `TTNNOpsAttrs.td` | Array of int64_t representing a shape |
| `ttnn::ShardSpec` | `TTNN_ShardSpecAttr` | `TTNNOpsAttrs.td` | CoreRangeSet + shape + orientation |
| `ttnn::NDShardSpec` | `TTNN_NDShardSpecAttr` | `TTNNOpsAttrs.td` | ND sharding extension of ShardSpec |

### C.2 Compute & Kernel Config Types

| TTNN C++ Type | MLIR Tablegen Type | Defined In | Notes |
|---|---|---|---|
| `ttnn::DeviceComputeKernelConfig` | `TTNN_DeviceComputeKernelConfig` | `TTNNOpsAttrs.td` | math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en. All params optional. |
| `ttnn::MathFidelity` | `TTNN_MathFidelity` (enum) | `TTNNOpsEnums.td` | LoFi, HiFi2, HiFi3, HiFi4 |
| `ttnn::operations::matmul::MatmulProgramConfig` | `AnyAttrOf<[TTNN_MatmulMultiCoreReuseProgramConfigAttr, TTNN_MatmulMultiCoreReuseMultiCastProgramConfigAttr, TTNN_MatmulMultiCoreReuseMultiCast1DProgramConfigAttr, TTNN_MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr]>` | `TTNNOpsAttrs.td` | **Quirky:** In C++, `MatmulProgramConfig` is a `std::variant` of 4 config types. In MLIR, this is modeled as `AnyAttrOf<[...]>` -- a union of 4 distinct attribute types. Each variant is a separate tablegen attr class with its own parameters. |
| `prim::LayerNormProgramConfig` | `TTNN_LayerNormShardedMultiCoreProgramConfigAttr` | `TTNNOpsAttrs.td` | Used in DistributedRMSNormOp. Not used in basic LayerNormOp/RMSNormOp (handled at runtime). |
| `ttnn::SDPAProgramConfig` | `TTNN_SDPAProgramConfigAttr` | `TTNNOpsAttrs.td` | For SDPA decode op. Contains compute_with_storage_grid_size, sub_core_grids, q/k_chunk_size, etc. |
| `Conv2dConfig` | `TTNN_Conv2dConfigAttr` | `TTNNOpsAttrs.td` | Complex config with many optional parameters for conv2d. |
| `Conv3dConfig` | `TTNN_Conv3dConfigAttr` | `TTNNOpsAttrs.td` | Config for 3D convolutions. |
| `Conv2dSliceConfig` | `TTNN_Conv2dSliceConfigAttr` | `TTNNOpsAttrs.td` | Slice type + num_slices for conv2d. |

### C.3 Data Type & Enum Types

| TTNN C++ Type | MLIR Tablegen Type | Defined In | Notes |
|---|---|---|---|
| `ttnn::DataType` / `tt::tt_metal::DataType` | `TTCore_DataTypeAttr` | `TTCoreOpsTypes.td` / `TTCoreOpsEnums.td` | Enum: Float32, Float16, BFloat16, BFP_Float8, BFP_BFloat8, UInt32, UInt16, UInt8, Int32, Bool, etc. |
| `tt::tt_fabric::Topology` | `TTCore_TopologyAttr` | `TTCoreOpsTypes.td` / `TTCoreOpsEnums.td` | Enum: Ring, Linear, Mesh, Torus, Disabled. Note: C++ uses `tt::tt_fabric::Topology`, MLIR uses `ttcore::Topology`. |
| `ttcore::ReduceType` | `TTCore_ReduceTypeAttr` | `TTCoreOpsEnums.td` | Enum: Sum, Mean, Max, Min, Std, Var, Prod, Invalid |
| `ttcore::MemorySpace` | `TTCore_MemorySpaceAttr` | `TTCoreOpsTypes.td` | Enum: System, SystemMMIO, DeviceDRAM, DeviceL1, RegisterDst |
| `ttcore::TensorMemoryLayout` | `TTCore_TensorMemoryLayoutAttr` | `TTCoreOpsTypes.td` | Enum: Interleaved, Sharded (note: different from TTNN enum which has Height/Width/BlockSharded) |
| `ttcore::MeshShardDirection` | `TTCore_MeshShardDirectionAttr` | `TTCoreOpsTypes.td` | Enum: FullToShard, ShardToFull |
| `ttcore::MeshShardType` | `TTCore_MeshShardTypeAttr` | `TTCoreOpsTypes.td` | Enum: Identity, Replicate, Maximal, Devices |
| `ttcore::OOBVal` | `TTCore_OOBValAttr` | `TTCoreOpsTypes.td` | Enum: Undef, Zero, One, Inf, NegInf |
| `ttcore::Arch` | `TTCore_ArchAttr` | `TTCoreOpsTypes.td` | Enum: WormholeB0, Blackhole |

### C.4 Device & Synchronization Types

| TTNN C++ Type | MLIR Tablegen Type | Defined In | Notes |
|---|---|---|---|
| `MeshDevice&` / `MeshDevice*` / `Device*` | `TTNN_Device` | `TTNNOpsTypes.td` | **Opaque type** (`!ttnn.device`). In C++, `MeshDevice` and `Device` are interchangeable references. In MLIR, there is a single opaque `TTNN_Device` type. It is produced by `TTNN_GetDeviceOp` and consumed by ops that need device access (Conv2dOp, ToDeviceOp, MeshShardOp, trace ops, etc.). The compiler does not inspect device internals at the IR level -- the runtime handles the mapping. |
| `tt::tt_metal::GlobalSemaphore` | `TTNN_GlobalSemaphore` | `TTNNOpsTypes.td` | **Opaque type** (`!ttnn.global_semaphore`). Created by `TTNN_CreateGlobalSemaphoreOp`, reset by `TTNN_ResetGlobalSemaphoreOp`. Used for cross-device synchronization in CCL ops. The semaphore's lifetime extends beyond individual op scope. |
| `tt::tt_metal::SubDeviceId` | `UI32Attr` | N/A (builtin) | In C++, `SubDeviceId` is a typed wrapper around `uint32_t`. In MLIR, it is modeled as a plain `UI32Attr` named `sub_device_id`. Used in CCL ops (AllGatherOp, ReduceScatterOp, AllReduceOp, DistributedRMSNormOp). |
| `CoreGrid` | Not directly mapped | N/A | In C++, matmul accepts `std::optional<const CoreGrid>`. This parameter is **not modeled** in the MLIR MatmulOp or LinearOp -- it is handled at runtime. See section F. |
| `CoreCoord` (TTNN) | `TTNN_CoreCoordAttr` | `TTNNOpsAttrs.td` | 2D coordinate (x, y) used in program configs. |
| `CoreRange` | `TTNN_CoreRangeAttr` | `TTNNOpsAttrs.td` | start_coord + end_coord |
| `CoreRangeSet` | `TTNN_CoreRangeSetAttr` | `TTNNOpsAttrs.td` | Array of non-intersecting CoreRange |
| `ttcore::GridAttr` | `TTCore_GridAttr` | `TTCoreOpsTypes.td` | Grid with shape (array of int64_t) and optional AffineMap mapping. Used in layout attrs. |
| `MeshShape` | `TTNN_MeshShapeAttr` | `TTNNOpsAttrs.td` | 2D coordinate (y, x) for mesh dimensions |
| `MeshOffset` | `TTNN_MeshOffsetAttr` | `TTNNOpsAttrs.td` | 2D coordinate (y, x) for mesh offset |

### C.5 Tile Type

| TTNN C++ Type | MLIR Tablegen Type | Defined In | Notes |
|---|---|---|---|
| `tt::tt_metal::Tile` | `TTCore_Tile` | `TTCoreOpsTypes.td` | A type (not an attr) representing a tile with shape [H, W] and DataType. Used as element type in tensors: `tensor<...x!ttcore.tile<32x32, bf16>>`. Default shape is 32x32. |

---

## D. Optional/Default Patterns

### D.1 Optional Parameters

| C++ Pattern | MLIR Tablegen Pattern | Example |
|---|---|---|
| `std::optional<T>` where T is an attr type | `OptionalAttr<ATTR_TYPE>` | `OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config` |
| `std::optional<T>` where T is a tensor | `Optional<AnyRankedTensor>` | `Optional<AnyRankedTensor>:$bias` |
| `std::optional<T>` where T is a device | `Optional<TTNN_Device>` | `Optional<TTNN_Device>:$device` (in ArangeOp) |
| `std::optional<uint32_t>` in an attr param | `OptionalParameter<"std::optional<uint32_t>">` | Inside attr class definitions (e.g., Conv2dConfigAttr params) |
| `std::optional<BoolAttr>` in an attr param | `OptionalParameter<"BoolAttr">` | Inside DeviceComputeKernelConfig params |
| `std::optional<DataType>` in an attr param | `OptionalParameter<"std::optional<ttcore::DataType>">` | Inside Conv2dConfigAttr, Conv3dConfigAttr |

### D.2 Default-Valued Parameters

| C++ Pattern | MLIR Tablegen Pattern | Example |
|---|---|---|
| `bool param = false` | `DefaultValuedAttr<BoolAttr, "false">` | `transpose_a`, `transpose_b`, `is_causal` |
| `bool param = true` | `DefaultValuedAttr<BoolAttr, "true">` | `reallocate_halo_output`, `count_include_pad` |
| `float epsilon = 1e-05` | `DefaultValuedAttr<F32Attr, "1e-05">` | `epsilon` in BatchNormOp |
| `float epsilon = 1e-12` | `DefaultValuedAttr<F32Attr, "1e-12">` | `epsilon` in RMSNormOp, LayerNormOp |
| `int dim = -1` | `DefaultValuedAttr<SI8Attr, "-1">` or `DefaultValuedAttr<I32Attr, "-1">` | `dim` in SortOp, TopKOp |
| `std::string mode = "none"` | `DefaultValuedAttr<StrAttr, "\"none\"">` | `approximate` in GeluBackwardOp |
| `std::string mode = "nearest"` | `DefaultValuedAttr<StrAttr, "\"nearest\"">` | `mode` in UpsampleOp |
| `std::string mode = "zeros"` | `DefaultValuedAttr<StrAttr, "\"zeros\"">` | `padding_mode` in Conv3dOp |
| `float pad_value = 0.0f` | `DefaultValuedOptionalAttr<F32Attr, "0.0f">` | `pad_value` in PermuteOp |
| `uint32_t cq_id = 0` | `DefaultValuedAttr<UI32Attr, "0">` | `cq_id` in WriteTensorOp, trace ops |

### D.3 DPS Output Tensor Handling

In TTNN C++, many ops accept `std::optional<Tensor> optional_output_tensor` for destination-passing style.
In MLIR, this is **generally NOT modeled as an op argument**. Instead:

- The compiler allocates output tensors via `ttnn.empty` ops separately.
- The result type on the op definition determines the output shape/type.
- At EmitC/runtime lowering, the optional output tensor is passed when needed.
- Exception: In-place ops (`TTNN_InplaceOp` subclass) like `UpdateCacheOp` explicitly have the output tensor as an `Arg<..., [MemWrite]>` input.

### D.4 AttrSizedOperandSegments

When an op has multiple `Optional<AnyRankedTensor>` operands, the trait `AttrSizedOperandSegments` must be
included. This tells MLIR to generate size tracking for each operand group. Examples:
- `TTNN_RMSNormOp` (optional weight, bias)
- `TTNN_BatchNormInferenceOp` (optional running_mean, running_var, weight, bias)
- `TTNN_ScaledDotProductAttentionDecodeOp` (optional attention_mask, cur_pos_tensor, attention_sink)

---

## E. Op Interfaces

### E.1 TTNN_MemoryConfigOpInterface

**File:** `include/ttmlir/Dialect/TTNN/Interfaces/TTNNTensorSpecInterface.td`

**Purpose:** Declares that an op has a `memory_config` attribute of type `TTNN_MemoryConfigAttr`.
Provides `getMemoryConfigAttr()` and `setMemoryConfigAttr()` methods. The optimizer uses this
interface to set/modify memory configurations.

**When to use:** Add to any op that accepts an optional or required `TTNN_MemoryConfigAttr` and
should participate in the optimizer's memory configuration pipeline.

**Usage in tablegen:**
```
def TTNN_MyOp : TTNN_Op<"my_op", [TTNN_MemoryConfigOpInterface]> {
    let arguments = (ins AnyRankedTensor:$input,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);
    ...
}
```

### E.2 TTNN_ComputeKernelConfigOpInterface

**File:** `include/ttmlir/Dialect/TTNN/Interfaces/TTNNTensorSpecInterface.td`

**Purpose:** Declares that an op has a `compute_config` attribute of type
`TTNN_DeviceComputeKernelConfig`. Provides `getComputeConfigAttr()` and `setComputeConfigAttr()`.

**When to use:** Add to any op that accepts `DeviceComputeKernelConfig` (compute-bound ops like
matmul, reductions, normalization, softmax, convolutions, CCL reduce_scatter).

**Usage in tablegen:**
```
def TTNN_MyOp : TTNN_Op<"my_op", [TTNN_ComputeKernelConfigOpInterface]> {
    let arguments = (ins AnyRankedTensor:$input,
                         OptionalAttr<TTNN_DeviceComputeKernelConfig>:$compute_config);
    ...
}
```

### E.3 TTNN_DtypeOpInterface

**File:** `include/ttmlir/Dialect/TTNN/Interfaces/TTNNTensorSpecInterface.td`

**Purpose:** Declares that an op has a `dtype` attribute of type `TTCore_DataTypeAttr`.
Provides `getDtypeAttr()` and `setDtypeAttr()`.

**When to use:** Add to ops where the output data type can differ from the input (e.g.,
binary ops, conv2d, embedding_bw, global_avg_pool2d, assign).

### E.4 TTNN_TensorSpecInterface

**File:** `include/ttmlir/Dialect/TTNN/Interfaces/TTNNTensorSpecInterface.td`

**Purpose:** Combines `TTNN_DtypeOpInterface`, `TTNN_LayoutOpInterface`, and
`TTNN_MemoryConfigOpInterface` into a single composite interface.

**When to use:** Used by creation ops (EmptyOp) that specify all layout properties.

### E.5 TTNN_LayoutOpInterface

**File:** `include/ttmlir/Dialect/TTNN/Interfaces/TTNNTensorSpecInterface.td`

**Purpose:** Declares that an op has a `layout` attribute of type `TTNN_LayoutAttr`.
Provides `getLayoutAttr()` and `setLayoutAttr()`.

### E.6 TTNN_DeviceOperandInterface

**File:** `include/ttmlir/Dialect/TTNN/Interfaces/TTNNDeviceOperandInterface.td`

**Purpose:** Interface for ops that have a device operand.

### E.7 TTNN_WorkaroundInterface

**File:** `include/ttmlir/Dialect/TTNN/Interfaces/TTNNWorkaroundInterface.td`

**Purpose:** Interface for ops that need operand workarounds (data type conversion, layout fixes).
Automatically included via `TTNN_Op` base class.

### E.8 TTNN_OpModelInterface

**File:** `include/ttmlir/Dialect/TTNN/Interfaces/TTNNOpModelInterface.td`

**Purpose:** Interface for ops that have op model constraints and runtime estimates.
Automatically included via `TTNN_Op` base class.

### E.9 TTNN_KernelInterface

**File:** `include/ttmlir/Dialect/TTNN/Interfaces/TTNNKernelInterface.td`

**Purpose:** Attribute interface for kernel descriptors in GenericOp programs. Provides methods
for getting symbol refs, core ranges, runtime args, and compile-time args.

---

## F. Types with No MLIR Equivalent

These C++ types from TTNN function signatures currently have **no MLIR representation** and are
handled at runtime only, or are omitted from the compiler's op model:

| C++ Type | Where Used | How Handled |
|---|---|---|
| `CoreGrid` (`std::optional<const CoreGrid>`) | matmul, linear | **Not modeled.** The MLIR matmul/linear ops do not include a `core_grid` parameter. The runtime/EmitC layer determines grid allocation. |
| `tt::tt_metal::Tile` (`std::optional<const Tile>`) | matmul, linear | **Not modeled as an op parameter.** Tile shape is encoded in the tensor's element type when using tiled layout (e.g., `!ttcore.tile<32x32, bf16>`), but the `output_tile` override parameter is not exposed in MLIR ops. |
| `GlobalCircularBuffer` (`std::optional<const GlobalCircularBuffer>`) | matmul, linear | **Not modeled.** The global circular buffer is a runtime optimization that is not represented in the IR. |
| `prim::LayerNormProgramConfig` | layer_norm, rms_norm | **Not modeled for basic LayerNorm/RMSNorm ops.** Only modeled as `TTNN_LayerNormShardedMultiCoreProgramConfigAttr` for `DistributedRMSNormOp`. For basic norm ops, the runtime selects the config automatically. |
| `Tensor residual_input_tensor` | layer_norm, rms_norm C++ | **Not modeled in basic norm ops.** Residual fusion is handled separately in `DistributedRMSNormOp`. |
| `Tensor recip_tensor` | layer_norm C++ | **Not modeled.** This is an internal optimization tensor. |
| `chunks_per_sync` | all_gather, reduce_scatter C++ | **Not modeled.** Runtime-only tuning parameter. |
| `num_workers_per_link` | all_gather, reduce_scatter C++ | **Not modeled.** Runtime-only tuning parameter. |
| `num_buffers_per_channel` | all_gather, reduce_scatter C++ | **Not modeled.** Runtime-only tuning parameter. |
| `sub_core_grid` (CoreRangeSet) | all_gather C++ | **Not modeled in AllGatherOp.** Only modeled in `TTNN_SDPAProgramConfigAttr`. |
| `intermediate_memory_config` | reduce_scatter C++ | **Not modeled.** Runtime handles intermediate allocation. |
| `float alpha, float beta` | addmm C++ | **Not modeled.** `addmm` is not a separate MLIR op; it decomposes to matmul + add. |
| `std::optional<const Activation>` (std::variant) | matmul/linear C++ | **Partially modeled** as `OptionalAttr<StrAttr>:$activation` in MatmulOp/LinearOp. The C++ type is `std::variant<std::string, UnaryWithParam>` but MLIR uses just a string. |

---

## G. Quick Reference: Common Op Argument Patterns

### Simple Elementwise Unary
```tablegen
let arguments = (ins AnyRankedTensor:$input,
                     OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);
let results = (outs AnyRankedTensor:$result);
```
Interfaces: `[TTNN_MemoryConfigOpInterface]`

### Elementwise Binary
```tablegen
let arguments = (ins AnyRankedTensor:$lhs,
                     AnyRankedTensor:$rhs,
                     OptionalAttr<TTCore_DataTypeAttr>:$dtype,
                     OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);
let results = (outs AnyRankedTensor:$result);
```
Interfaces: `[TTNN_MemoryConfigOpInterface, TTNN_DtypeOpInterface]`

### Reduction Op
```tablegen
let arguments = (ins AnyRankedTensor:$input,
                     BoolAttr:$keep_dim,
                     OptionalAttr<I32ArrayAttr>:$dim_arg,
                     OptionalAttr<TTNN_DeviceComputeKernelConfig>:$compute_config);
let results = (outs AnyRankedTensor:$result);
```
Interfaces: `[TTNN_ComputeKernelConfigOpInterface]`

### Normalization Op (with optional weight/bias)
```tablegen
let arguments = (ins AnyRankedTensor:$input,
                     Optional<AnyRankedTensor>:$weight,
                     Optional<AnyRankedTensor>:$bias,
                     DefaultValuedAttr<F32Attr, "1e-12">:$epsilon,
                     OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config,
                     OptionalAttr<TTNN_DeviceComputeKernelConfig>:$compute_config);
let results = (outs AnyRankedTensor:$result);
```
Traits: `[AttrSizedOperandSegments]` (because of multiple Optional tensors)
Interfaces: `[TTNN_MemoryConfigOpInterface, TTNN_ComputeKernelConfigOpInterface]`

### CCL Op
```tablegen
let arguments = (ins AnyRankedTensor:$input,
                     SI32Attr:$dim,
                     UI32Attr:$cluster_axis,
                     OptionalAttr<UI32Attr>:$sub_device_id,
                     OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config,
                     OptionalAttr<UI32Attr>:$num_links,
                     OptionalAttr<TTCore_TopologyAttr>:$topology);
let results = (outs AnyRankedTensor:$result);
```
Interfaces: `[TTNN_MemoryConfigOpInterface]`

### Convolution Op (requires device)
```tablegen
let arguments = (ins AnyRankedTensor:$input,
                     AnyRankedTensor:$weight,
                     Optional<AnyRankedTensor>:$bias,
                     TTNN_Device:$device,
                     I32Attr:$in_channels,
                     I32Attr:$out_channels,
                     ...
                     OptionalAttr<TTCore_DataTypeAttr>:$dtype,
                     OptionalAttr<TTNN_Conv2dConfigAttr>:$conv2d_config,
                     OptionalAttr<TTNN_DeviceComputeKernelConfig>:$compute_config);
let results = (outs AnyRankedTensor:$result);
```
Interfaces: `[TTNN_DtypeOpInterface, TTNN_ComputeKernelConfigOpInterface]`

---

## H. MatmulOp Program Config Details

The `MatmulOp` and `LinearOp` have a notable pattern for handling `MatmulProgramConfig`:

```tablegen
OptionalAttr<AnyAttrOf<[
   TTNN_MatmulMultiCoreReuseProgramConfigAttr,
   TTNN_MatmulMultiCoreReuseMultiCastProgramConfigAttr,
   TTNN_MatmulMultiCoreReuseMultiCast1DProgramConfigAttr,
   TTNN_MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr
]>>:$matmul_program_config
```

This uses `AnyAttrOf<[...]>` which allows any one of the four config attribute types.
In C++, this corresponds to `std::variant<MatmulMultiCoreReuseProgramConfig, ...>` wrapped
in `using MatmulProgramConfig = std::variant<...>`.

Each config variant has different parameters:
- **MatmulMultiCoreReuse**: compute_with_storage_grid_size, in0_block_w, out_subblock_h/w, per_core_m/n
- **MatmulMultiCoreReuseMultiCast**: Same as above + out_block_h/w, transpose_mcast, fused_activation, fuse_batch
- **MatmulMultiCoreReuseMultiCast1D**: Same as MultiCast + mcast_in0, gather_in0, hop_cores, num_global_cb_receivers, untilize_out
- **MatmulMultiCoreReuseMultiCastDRAMSharded**: in0_block_w, per_core_m/n, fused_activation

All config variants include `CoreCoordAttr` for `compute_with_storage_grid_size` (except DRAMSharded).

---

## I. Key Files Summary

| File | Contents |
|---|---|
| `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td` | All TTNN op definitions |
| `include/ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.td` | Custom TTNN attribute types (MemoryConfig, program configs, kernel attrs, etc.) |
| `include/ttmlir/Dialect/TTNN/IR/TTNNOpsEnums.td` | TTNN-specific enums (Layout, TensorMemoryLayout, BufferType, MathFidelity, UnaryOpType, etc.) |
| `include/ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.td` | TTNN opaque types (TTNN_Device, TTNN_GlobalSemaphore) |
| `include/ttmlir/Dialect/TTNN/IR/TTNNBase.td` | Base classes: TTNN_Op, TTNN_InplaceOp, TTNN_MemFreeOp, TTNN_MemoryEffectOp |
| `include/ttmlir/Dialect/TTNN/IR/TTNNTraits.td` | TTNN-specific traits |
| `include/ttmlir/Dialect/TTNN/Interfaces/TTNNTensorSpecInterface.td` | Op interfaces: MemoryConfigOp, ComputeKernelConfigOp, DtypeOp, LayoutOp, TensorSpec |
| `include/ttmlir/Dialect/TTNN/Interfaces/TTNNDeviceOperandInterface.td` | Device operand interface |
| `include/ttmlir/Dialect/TTNN/Interfaces/TTNNKernelInterface.td` | Kernel attribute interface |
| `include/ttmlir/Dialect/TTCore/IR/TTCoreOpsEnums.td` | Shared enums: DataType, Arch, MemorySpace, ReduceType, Topology, MeshShard*, etc. |
| `include/ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td` | Shared types: GridAttr, TileType, DeviceAttr, SystemDescAttr, DataTypeAttr, TopologyAttr, etc. |
