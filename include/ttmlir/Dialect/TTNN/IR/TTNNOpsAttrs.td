// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSATTRS_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSATTRS_TD

include "ttmlir/Dialect/TTNN/IR/TTNNBase.td"
include "ttmlir/Dialect/TTNN/IR/TTNNOpsEnums.td"
include "ttmlir/Dialect/TTNN/Interfaces/TTNNKernelInterface.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

//===----------------------------------------------------------------------===//
// TTNN attr definitions
//===----------------------------------------------------------------------===//

class TTNN_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Attribute">
    : AttrDef<TTNN_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
  let attrName = "ttnn." # attrMnemonic;
}

class TTNN_Mesh2DCoordAttr<string name, string mnemonic> : TTNN_Attr<name, mnemonic> {
  let parameters = (ins "int64_t":$y, "int64_t":$x);
  let assemblyFormat = "custom<VargDimensionList>($y, $x)";
}

def TTNN_CoreCoordAttr : TTNN_Attr<"CoreCoord", "core_coord"> {
  let summary = "A 2D coordinate of a core in the core grid";
  let description = [{
    Specifies a coordinate representing a specific core

    Parameters:
    - `x`: The x-coordinate in the core grid
    - `y`: The y-coordinate in the core grid

    Example:

    ```mlir
    #core = #ttnn.core_coord<3, 4>
    ```
  }];

  let parameters = (ins "uint64_t":$x, "uint64_t":$y);
  let assemblyFormat = "`<` params `>`";
}

def TTNN_CoreRangeAttr: TTNN_Attr<"CoreRange", "core_range"> {
  let summary = "A range of cores in the core grid";
  let description = [{
    Defines a range of cores in the core grid.

    Parameters:
    - `start_coord`: Lower-left corner of the core range
    - `end_coord`: Upper-right corner of the core range

    Constraints:
    - The condition `start_coord` <= `end_coord` must be satisfied

    Example:

    ```mlir
    // Define a core range spanning from (1,1) to (3,4)
    // This represents a 3x4 rectangular grid of cores
    #core_range = #ttnn.core_range<(1, 1), (3, 4)>
    ```
  }];

  let parameters = (ins "CoreCoordAttr":$start_coord, "CoreCoordAttr":$end_coord);
  let assemblyFormat = "`<` custom<CoordBracketStyle>($start_coord) `,` custom<CoordBracketStyle>($end_coord) `>`";

  let extraClassDeclaration = [{
    bool intersects(CoreRangeAttr other) const;
  }];

  let genVerifyDecl = 1;
}

def TTNN_CoreRangeSetAttr: TTNN_Attr<"CoreRangeSet", "core_range_set"> {
  let summary = "Represents a set of core ranges in the core grid";
  let description = [{
    Defines a collection of core ranges within the core grid.
    No two core ranges in the set should intersect.

    Parameters:
    - `core_ranges`: An array of core ranges

    Constraints:
    - The set must contain non-intersecting core ranges
    - Can be empty (no ranges)

    Example:

    ```mlir
    // Define empty core range set
    #empty_core_range_set = #ttnn.core_range_set<>

    // Define three non-intersecting core ranges
    #core_range_set = #ttnn.core_range_set<[
      #ttnn.core_range<(0, 0), (2, 2)>,
      #ttnn.core_range<(3, 3), (5, 5)>,
      #ttnn.core_range<(6, 0), (7, 4)>
    ]>
    ```
  }];

  let parameters = (ins OptionalArrayRefParameter<"CoreRangeAttr">:$core_ranges);
  let assemblyFormat = "`<` (`[` qualified($core_ranges)^ `]`)? `>`";

  let genVerifyDecl = 1;
}

def TTNN_LayoutAttr : EnumAttr<TTNN_Dialect, TTNN_Layout, "layout"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTNN_TensorMemoryLayoutAttr : EnumAttr<TTNN_Dialect, TTNN_TensorMemoryLayout, "tensor_memory_layout"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTNN_BufferTypeAttr : EnumAttr<TTNN_Dialect, TTNN_BufferType, "buffer_type"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTNN_ShapeAttr : TTNN_Attr<"Shape", "shape"> {
  let summary = "TTNN Shape attribute";
  let description = [{
    TTNN shape attribute
  }];

  let parameters = (ins ArrayRefParameter<"int64_t">:$shape);
  let assemblyFormat = "`<` custom<DimensionList>($shape) `>`";
}

def TTNN_ShardOrientationAttr : EnumAttr<TTNN_Dialect, TTNN_ShardOrientation, "shard_orientation"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTNN_ShardModeAttr : EnumAttr<TTNN_Dialect, TTNN_ShardMode, "shard_mode"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTNN_ShardSpecAttr : TTNN_Attr<"ShardSpec", "shard_spec"> {
  let summary = "TTNN ShardSpec attribute";
  let description = [{
    The `ShardSpecAttr` defines how a tensor is sharded across multiple cores in the Tenstorrent hardware.

    This attribute specifies:
    - Which cores the tensor is distributed across (via CoreRangeSet)
    - The shape of each shard
    - The orientation of the sharding (row_major, column_major, etc.)
    - The sharding mode (physical, logical)
    - The physical shard shape that is only present if the sharding mode isn't physical

    ShardSpec is used within MemoryConfigAttr to specify sharding information for tensors
    stored in L1 memory with sharded memory layouts (block_sharded, width_sharded, or height_sharded).

    Example:
    ```mlir
    #ttnn.shard_spec<#ttnn.core_range_set<[#ttnn.core_range<(0,0), (7, 0)>]>, <32x128>, <row_major>, <physical>>
    #ttnn.shard_spec<#ttnn.core_range_set<[#ttnn.core_range<(0,0), (7, 0)>]>, <32x128>, <row_major>, <logical>, <32x128>>
    ```

    This attribute is optional in the MemoryConfigAttr and should only be present
    when the tensor memory layout is one of the sharded layouts.
  }];

  let parameters = (ins "CoreRangeSetAttr": $coreRangeSet,
                    "ShapeAttr":$shape,
                    "ShardOrientationAttr":$shardOrientation,
                    "ShardModeAttr":$shardMode,
                    OptionalParameter<"ShapeAttr">:$physical_shard_shape);

  let builders =
    [
      AttrBuilder<(ins "ShapeAttr": $shape, "ttcore::GridAttr": $shardGrid, "ttcore::GridAttr": $deviceGrid),
      [{
        CoreRangeSetAttr coreRangeSet = getCoreRangeSet($_ctxt, shardGrid, deviceGrid);
        return Base::get($_ctxt, coreRangeSet, shape, ShardOrientationAttr::get($_ctxt, ::mlir::tt::ttnn::ShardOrientation::RowMajor), ShardModeAttr::get($_ctxt, ::mlir::tt::ttnn::ShardMode::Physical), ShapeAttr());
      }]>,
      AttrBuilder<(ins "TTNNLayoutAttr": $layout, "ttcore::GridAttr": $deviceGrid),
      [{
        ttcore::GridAttr shardGrid = layout.getGrid();
        CoreRangeSetAttr coreRangeSet = getCoreRangeSet($_ctxt, shardGrid, deviceGrid);
        return Base::get($_ctxt, coreRangeSet, ShapeAttr::get($_ctxt,layout.getScalarShardShape()), ShardOrientationAttr::get($_ctxt, ::mlir::tt::ttnn::ShardOrientation::RowMajor), ShardModeAttr::get($_ctxt, ::mlir::tt::ttnn::ShardMode::Physical), ShapeAttr());
      }]>
    ];

  let assemblyFormat = "`<` params `>`";
  let extraClassDeclaration = [{
    static CoreRangeSetAttr getCoreRangeSet(mlir::MLIRContext *context, ttcore::GridAttr shardGrid, ttcore::GridAttr deviceGrid);
  }];
}

def TTNN_MemoryConfigAttr : TTNN_Attr<"MemoryConfig", "memory_config"> {
  let summary = "TTNN MemoryConfig attribute";
  let description = [{
    The `MemoryConfigAttr` defines how a tensor is stored in memory on Tenstorrent hardware.

    This attribute specifies:
    - `bufferType` - specifies which memory type to use (L1, DRAM, System Memory).
    - `tensorMemoryLayout` - defines how the tensor is laid out in memory (interleaved, block_sharded, width_sharded, height_sharded)
    - `shardSpec` - optional parameter is only used with sharded memory layouts and defines how the tensor is distributed across multiple cores.

    Examples:
    ```mlir
    // Simple interleaved memory in DRAM
    #ttnn.memory_config<#dram, <interleaved>>

    // L1 memory with block sharding across cores
    #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<#ttnn.core_range_set<[#ttnn.core_range<(0,0), (7, 0)>]>, <32x128>, <row_major>, <physical>>>
    ```
  }];

  let parameters = (ins OptionalParameter<"TensorMemoryLayoutAttr">:$tensorMemoryLayout,
                    "BufferTypeAttr":$bufferType,
                    OptionalParameter<"std::optional<ShardSpecAttr>">:$shardSpec);

  let assemblyFormat = "`<` $bufferType (`,` $tensorMemoryLayout^ )? (`,` $shardSpec^ )? `>`";

  let extraClassDeclaration = [{
    class Builder;
    static MemoryConfigAttr get(ttnn::TTNNLayoutAttr layout, ttcore::GridAttr deviceGrid);
    llvm::ArrayRef<int64_t> getShardShape(bool convertTileToScalar = true) const;
  }];

  let genVerifyDecl = 1;
}

def UnaryWithParamAttr : TTNN_Attr<"UnaryWithParam", "unary_with_param"> {
  let summary = "A unary operation with parameters";
  let description = [{
    Defines a unary operation with additional parameters, used
    for fused activations.

    Parameters:
    - `op_type`: Type of unary operation
    - `params`: Optional parameters

    Example:

    ```mlir
    // Unary operation with no parameters
    #unary_witout_params = #ttnn.unary_with_param<op_type = relu>

    // Unary operation with parameters
    #unary_with_params = #ttnn.unary_with_param<op_type = add_unary_sfpu, params = [1.0 : f32]>
    ```
  }];

  let parameters = (ins "UnaryOpType":$op_type,
                        OptionalArrayRefParameter<"FloatAttr">:$params);

  let assemblyFormat = "`<` `op_type` `=` qualified($op_type) (`,` `params` `=` `[` $params^ `]`)? `>`";
}

def TTNN_MatmulMultiCoreReuseProgramConfigAttr : TTNN_Attr<"MatmulMultiCoreReuseProgramConfig", "matmul_multi_core_reuse_program_config"> {
  let summary = "TTNN MatmulMultiCoreReuseProgramConfig";
  let description = [{
    TTNN MatmulMultiCoreReuseProgramConfig

    Example:

    ```mlir
    #matmul_program_config = #ttnn.matmul_multi_core_reuse_program_config<
      compute_with_storage_grid_size = #ttnn.core_coord<7, 9>,
      in0_block_w = 8,
      out_subblock_h = 1,
      out_subblock_w = 8,
      per_core_m = 8,
      per_core_n = 8
    >
    ```
  }];

  let parameters = (ins "CoreCoordAttr":$compute_with_storage_grid_size,
                        "uint64_t":$in0_block_w,
                        "uint64_t":$out_subblock_h,
                        "uint64_t":$out_subblock_w,
                        "uint64_t":$per_core_m,
                        "uint64_t":$per_core_n);

  let assemblyFormat = [{
    `<` struct(
      qualified($compute_with_storage_grid_size),
      $in0_block_w,
      $out_subblock_h,
      $out_subblock_w,
      $per_core_m,
      $per_core_n
    ) `>`
  }];
}

def TTNN_MatmulMultiCoreReuseMultiCastProgramConfigAttr : TTNN_Attr<"MatmulMultiCoreReuseMultiCastProgramConfig", "matmul_multi_core_reuse_multi_cast_program_config"> {
  let summary = "TTNN MatmulMultiCoreReuseMultiCastProgramConfig";
  let description = [{
    TTNN MatmulMultiCoreReuseMultiCastProgramConfig

    Example:

    ```mlir
    #matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<
      compute_with_storage_grid_size = #ttnn.core_coord<8, 8>,
      in0_block_w = 16,
      out_subblock_h = 2,
      out_subblock_w = 4,
      out_block_h = 2,
      out_block_w = 4,
      per_core_m = 2,
      per_core_n = 4,
      transpose_mcast = true,
      fused_activation = #ttnn.unary_with_param<op_type = relu>,
      fuse_batch = true
    >
    ```
  }];

  let parameters = (ins "CoreCoordAttr":$compute_with_storage_grid_size,
                        "uint64_t":$in0_block_w,
                        "uint64_t":$out_subblock_h,
                        "uint64_t":$out_subblock_w,
                        "uint64_t":$out_block_h,
                        "uint64_t":$out_block_w,
                        "uint64_t":$per_core_m,
                        "uint64_t":$per_core_n,
                        "bool":$transpose_mcast,
                        OptionalParameter<"UnaryWithParamAttr">:$fused_activation,
                        "bool":$fuse_batch);

  let assemblyFormat = [{
    `<` struct(
      qualified($compute_with_storage_grid_size),
      $in0_block_w,
      $out_subblock_h,
      $out_subblock_w,
      $out_block_h,
      $out_block_w,
      $per_core_m,
      $per_core_n,
      $transpose_mcast,
      qualified($fused_activation),
      $fuse_batch
    ) `>`
  }];
}

def TTNN_MatmulMultiCoreReuseMultiCast1DProgramConfigAttr : TTNN_Attr<"MatmulMultiCoreReuseMultiCast1DProgramConfig", "matmul_multi_core_reuse_multi_cast_1d_program_config"> {
  let summary = "TTNN MatmulMultiCoreReuseMultiCast1DProgramConfig";
  let description = [{
    TTNN MatmulMultiCoreReuseMultiCast1DProgramConfig

    Example:

    ```mlir
    #matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<
      compute_with_storage_grid_size = #ttnn.core_coord<8, 2>,
      in0_block_w = 32,
      out_subblock_h = 1,
      out_subblock_w = 2,
      out_block_h = 8,
      out_block_w = 6,
      per_core_m = 8,
      per_core_n = 6,
      fuse_batch = true,
      fused_activation = #ttnn.unary_with_param<op_type = add_unary_sfpu, params = [1.0 : f32]>,
      mcast_in0 = true,
      gather_in0 = false,
      hop_cores = #ttnn.core_range_set<>,
      num_global_cb_receivers = 0,
      untilize_out = false
    >
    ```
  }];

  let parameters = (ins "CoreCoordAttr":$compute_with_storage_grid_size,
                        "uint64_t":$in0_block_w,
                        "uint64_t":$out_subblock_h,
                        "uint64_t":$out_subblock_w,
                        "uint64_t":$out_block_h,
                        "uint64_t":$out_block_w,
                        "uint64_t":$per_core_m,
                        "uint64_t":$per_core_n,
                        "bool":$fuse_batch,
                        OptionalParameter<"UnaryWithParamAttr">:$fused_activation,
                        "bool":$mcast_in0,
                        "bool":$gather_in0,
                        "CoreRangeSetAttr":$hop_cores,
                        "uint64_t":$num_global_cb_receivers,
                        "bool":$untilize_out);

  let assemblyFormat = [{
    `<` struct(
      qualified($compute_with_storage_grid_size),
      $in0_block_w,
      $out_subblock_h,
      $out_subblock_w,
      $out_block_h,
      $out_block_w,
      $per_core_m,
      $per_core_n,
      $fuse_batch,
      qualified($fused_activation),
      $mcast_in0,
      $gather_in0,
      qualified($hop_cores),
      $num_global_cb_receivers,
      $untilize_out
    ) `>`
  }];
}

def TTNN_MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr : TTNN_Attr<"MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig", "matmul_multi_core_reuse_multi_cast_dram_sharded_program_config"> {
  let summary = "TTNN MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig";
  let description = [{
    TTNN MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig

    Example:

    ```mlir
    #matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_dram_sharded_program_config<
      in0_block_w = 1,
      per_core_m = 1,
      per_core_n = 5,
      fused_activation = #ttnn.unary_with_param<op_type = relu>
    >
    ```
  }];

  let parameters = (ins "uint64_t":$in0_block_w,
                        "uint64_t":$per_core_m,
                        "uint64_t":$per_core_n,
                        OptionalParameter<"UnaryWithParamAttr">:$fused_activation);

  let assemblyFormat = "`<` struct($in0_block_w, $per_core_m, $per_core_n, qualified($fused_activation)) `>`";
}

def TTNN_Conv2dConfigAttr : TTNN_Attr<"Conv2dConfig", "conv2d_config"> {
  let summary = "TTNN Conv2dConfig attribute";
  let description = [{
    Configuration parameters for TTNN conv2d operations that control memory usage,
    performance optimizations, and execution behavior.

    Parameters:
    - `weights_dtype`: Data type for weights and bias tensor after preprocessing (default: bfloat16)
    - `activation`: Optional activation function to fuse ("relu" or "")
    - `deallocate_activation`: Whether to deallocate input activation tensor memory (default: false)
    - `reallocate_halo_output`: Whether to reallocate intermediate halo tensor to reduce memory fragmentation (default: false)
    - `act_block_h_override`: Override for act_block_h parameter (must be multiple of 32). Smaller values reduce memory usage but decrease performance. Ignored when shard_layout = WIDTH_SHARDED (default: 0)
    - `act_block_w_div`: Divisor for maximum possible act_block_w parameter. Only useful when in_channels > 2048. Ignored when shard_layout = HEIGHT_SHARDED or BLOCK_SHARDED (default: 1)
    - `reshard_if_not_optimal`: Whether operation can re-shard input tensor for optimal performance. Mutually exclusive with override_sharding_config (default: false)
    - `override_sharding_config`: Whether to override input sharding config with provided shard_layout. Mutually exclusive with reshard_if_not_optimal (default: false)
    - `shard_layout`: Optional tensor memory layout for sharding type specification
    - `core_grid`: Optional core grid specification. Only applicable when override_sharding_config = true
    - `transpose_shards`: Whether shards should be distributed in ROW_MAJOR order. Only applicable when not using height sharding (default: true)
    - `output_layout`: Layout of output tensor - TILE_LAYOUT or ROW_MAJOR_LAYOUT (default: TILE_LAYOUT)
    - `enable_act_double_buffer`: Enable activation double buffering for increased performance at cost of higher L1 usage (default: false)
    - `enable_weights_double_buffer`: Enable weights double buffering when using block sharding for increased performance at cost of higher L1 usage (default: false)
    - `in_place`: Re-use input tensor storage when creating output tensor (default: false)

    Example:

    ```mlir
    #conv2d_config = #ttnn.conv2d_config<
      weights_dtype = bfloat16,
      activation = "relu",
      deallocate_activation = false,
      reallocate_halo_output = false,
      act_block_h_override = 64,
      act_block_w_div = 1,
      reshard_if_not_optimal = false,
      override_sharding_config = false,
      shard_layout = block_sharded,
      transpose_shards = true,
      output_layout = tile,
      enable_act_double_buffer = false,
      enable_weights_double_buffer = false,
      in_place = false
    >
    ```
  }];

  let parameters = (ins OptionalParameter<"std::optional<ttcore::DataType>">:$weights_dtype,
                        OptionalParameter<"UnaryWithParamAttr">:$activation,
                        OptionalParameter<"BoolAttr">:$deallocate_activation,
                        OptionalParameter<"BoolAttr">:$reallocate_halo_output,
                        OptionalParameter<"std::optional<uint32_t>">:$act_block_h_override,
                        OptionalParameter<"std::optional<uint32_t>">:$act_block_w_div,
                        OptionalParameter<"BoolAttr">:$reshard_if_not_optimal,
                        OptionalParameter<"BoolAttr">:$override_sharding_config,
                        OptionalParameter<"std::optional<TensorMemoryLayout>">:$shard_layout,
                        OptionalParameter<"CoreRangeSetAttr">:$core_grid,
                        OptionalParameter<"BoolAttr">:$transpose_shards,
                        OptionalParameter<"std::optional<Layout>">:$output_layout,
                        OptionalParameter<"BoolAttr">:$enable_act_double_buffer,
                        OptionalParameter<"BoolAttr">:$enable_weights_double_buffer,
                        OptionalParameter<"BoolAttr">:$in_place);

  let builders = [
    AttrBuilder<(ins )>
  ];

  let extraClassDeclaration = [{
    static Conv2dConfigAttr getDefault(::mlir::MLIRContext *context);

    Conv2dConfigAttr withActivation(UnaryOpType activation) const;
    Conv2dConfigAttr withWeightsDtype(ttcore::DataType dtype) const;
    Conv2dConfigAttr withDeallocateActivation(bool value) const;
    Conv2dConfigAttr withReallocateHaloOutput(bool value) const;
    Conv2dConfigAttr withActBlockHOverride(uint32_t value) const;
    Conv2dConfigAttr withActBlockWDiv(uint32_t value) const;
    Conv2dConfigAttr withReshardIfNotOptimal(bool value) const;
    Conv2dConfigAttr withOverrideShardingConfig(bool value) const;
    Conv2dConfigAttr withShardLayout(TensorMemoryLayout layout) const;
    Conv2dConfigAttr withCoreGrid(CoreRangeSetAttr grid) const;
    Conv2dConfigAttr withTransposeShards(bool value) const;
    Conv2dConfigAttr withOutputLayout(Layout layout) const;
    Conv2dConfigAttr withEnableActDoubleBuffer(bool value) const;
    Conv2dConfigAttr withEnableWeightsDoubleBuffer(bool value) const;
    Conv2dConfigAttr withInPlace(bool value) const;
    bool hasActivation() const;
    bool hasWeightsDtype() const;
    bool hasDeallocateActivation() const;
    bool hasReallocateHaloOutput() const;
    bool hasActBlockHOverride() const;
    bool hasActBlockWDiv() const;
    bool hasReshardIfNotOptimal() const;
    bool hasOverrideShardingConfig() const;
    bool hasShardLayout() const;
    bool hasCoreGrid() const;
    bool hasTransposeShards() const;
    bool hasOutputLayout() const;
    bool hasPreprocessWeightsOnDevice() const;
    bool hasAlwaysPreprocessWeights() const;
    bool hasEnableActDoubleBuffer() const;
    bool hasEnableWeightsDoubleBuffer() const;
    bool hasInPlace() const;
  }];

  let assemblyFormat = "`<` struct(params) `>`";
}

def TTNN_DeviceComputeKernelConfig : TTNN_Attr<"DeviceComputeKernelConfig", "device_compute_kernel_config"> {
  let summary = "TTNN DeviceComputeKernelConfig attribute";
  let description = [{
    The TTNN_DeviceComputeKernelConfig attribute configures compute kernel execution parameters for tensor operations on Tenstorrent devices. This attribute provides fine-grained control over mathematical precision, memory usage, and synchronization behavior during compute operations.

    Parameters:
      - `math_fidelity`: Controls the mathematical precision and accuracy of compute operations. This parameter affects the trade-off between computational speed and numerical precision. Higher fidelity modes provide more accurate results but may require additional computational cycles.
      - `math_approx_mode`: Configures SFPU operation mode:
        - Precise mode (false): Higher accuracy with more computational cycles and better PCC
        - Approximate mode (true): Faster execution with fewer cycles but reduced accuracy
      - `fp32_dest_acc_en`: Configures destination registers to use 32-bit floating-point precision instead of the default 16-bit mode. It provides higher precision at the cost of reducing available destination register count by half.
      - `packer_l1_acc`: When packing multiple tiles to the same address, subsequent packs perform accumulation (addition using FP16 or FP32 precision) rather than overwriting.
      - `dst_full_sync_en`: Configures destination register acquisition mode:
        - Half mode (false): Acquires 8 tiles in destination registers
        - Full mode (true): Acquires 16 tiles in destination registers, providing increased parallelism at the cost of higher resource usage

    Example:

    ```mlir
    #device_compute_kernel_config = #ttnn.device_compute_kernel_config<
      math_fidelity = lofi,
      math_approx_mode = true,
      fp32_dest_acc_en = false,
      packer_l1_acc = false,
      dst_full_sync_en = false
    >
    ```
  }];

  let parameters = (ins OptionalParameter<"std::optional<MathFidelity>">:$math_fidelity,
                        OptionalParameter<"BoolAttr">:$math_approx_mode,
                        OptionalParameter<"BoolAttr">:$fp32_dest_acc_en,
                        OptionalParameter<"BoolAttr">:$packer_l1_acc,
                        OptionalParameter<"BoolAttr">:$dst_full_sync_en);

  let builders = [
    AttrBuilder<(ins ),
    [{
      return Base::get($_ctxt, std::nullopt, nullptr, nullptr, nullptr, nullptr);
    }]>,
  ];

  let extraClassDeclaration = [{
    DeviceComputeKernelConfigAttr withMathFidelity(MathFidelity mathFidelity) const;
    DeviceComputeKernelConfigAttr withMathApproxMode(bool value) const;
    DeviceComputeKernelConfigAttr withFp32DestAccEn(bool value) const;
    DeviceComputeKernelConfigAttr withPackerL1Acc(bool value) const;
    DeviceComputeKernelConfigAttr withDstFullSyncEn(bool value) const;
  }];

  let assemblyFormat = "`<` struct(params) `>`";
}

def TTNN_MeshShapeAttr : TTNN_Mesh2DCoordAttr<"MeshShape", "mesh_shape"> {
  let summary = "TTNN Mesh Shape";
  let description = [{
    TTNN mesh shape representing the dimensions of a 2D mesh.
  }];
}

def TTNN_MeshOffsetAttr : TTNN_Mesh2DCoordAttr<"MeshOffset", "mesh_offset"> {
  let summary = "TTNN Mesh Offset";
  let description = [{
    TTNN mesh offset representing the starting coordinates in a 2D mesh.
  }];
}

def TTNN_TTNNLayoutAttr: TTNN_Attr<"TTNNLayout", "ttnn_layout"> {
  let summary = "Tensor encoding attribute used for types in ttnn";
  let description = [{
    Layout attribute in ttnn. This attribute is used to encode different information about tensor memory layout.
    Here is how tensor will look like after layout tensor<32x32x64xf32, #ttnn.ttnn_layout<linear, grid, memref, mem_layout>>
    Lets break down what each parameter means:
    - linear: An affine map that defines how the logical tensor dimensions map to physical space.
    - grid: The grid shape (of tensix cores) where tensor is divided onto. For non-L1 buffer type, grid shape has to be 1x1.
    - memref: A memref is used to describe shard size and memory space. Shard size is calculated by dividing the tensor size by grid size.
    - mem_layout: The layout of the tensor in memory. For tensor on host it should be None. For tensor on device
    it can be interleaved or sharded.
    - mesh_sharding: The mesh of the tensor in multi-devices.
  }];

  let parameters = (ins AttrParameter<"AffineMap", "An affine map that defines how the logical tensor dimensions map to a grid shape.">:$linear,
                        AttrParameter<"ttcore::GridAttr", "The grid shape that this tensor is divided onto.">:$grid,
                        AttrParameter<"MemRefType", "A memref that describes the physical footprint allocation of the shard. It must also have a shape with rank equal to grid.">:$memref,
                        OptionalParameter<"TensorMemoryLayoutAttr", "TTNN tensor memory layout">:$mem_layout,
                        OptionalParameter<"ttcore::TensorMeshAttr", "TT tensor mesh attr">:$tensor_mesh,
                        OptionalParameter<"bool", "A status flag, asking the users to ignore the physical layout. This is used to model a sharded layout with unspecified shard shape.">:$ignorePhysicalLayout);
  let assemblyFormat = "`<` $linear`,` $grid`,` (`mesh` `=` $tensor_mesh^ `,`)? $memref (`,` $mem_layout^)? (`,` $ignorePhysicalLayout^)? `>`";
  let extraClassDeclaration = [{
    static TTNNLayoutAttr get(::mlir::MLIRContext *context,
                        AffineMap linear,
                        ttcore::GridAttr grid, MemRefType memref,
                        TensorMemoryLayoutAttr mem_layout,
                        ttcore::TensorMeshAttr tensor_mesh);

    static TTNNLayoutAttr get(::mlir::MLIRContext *context,
                        ArrayRef<int64_t> tensorShape,
                        Type elementType,
                        BufferType bufferType,
                        ttcore::GridAttr grid,
                        TensorMemoryLayoutAttr memoryLayoutAttr = nullptr,
                        ttcore::TensorMeshAttr tensorMeshAttr = nullptr,
                        ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals = {{0, -1}},
                        bool ignorePhysicalLayout = false);

    TTNNLayoutAttr withGrid(ArrayRef<int64_t> tensorShape, ttcore::GridAttr grid, ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals = {{0, -1}});
    TTNNLayoutAttr withGrid(RankedTensorType ty,
                        ttcore::GridAttr grid,
                        ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals = {{0, -1}});
    TTNNLayoutAttr withElementType(Type elementType, ArrayRef<int64_t> tensorShape, ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals = {{0, -1}});
    TTNNLayoutAttr withBufferType(BufferType bufferType);
    TTNNLayoutAttr withMemoryLayout(TensorMemoryLayoutAttr memLayoutAttr);
    TTNNLayoutAttr withMemoryLayout(TensorMemoryLayout memLayout);
    TTNNLayoutAttr withLayout(Layout layout, ArrayRef<int64_t> tensorShape);
    TTNNLayoutAttr withShardShape(llvm::SmallVector<int64_t> shardShape);
    TTNNLayoutAttr withTensorShape(ArrayRef<int64_t> tensorShape);
    TTNNLayoutAttr withIgnorePhysicalLayout(bool ignorePhysicalLayout);
    TTNNLayoutAttr withDataType(ttcore::DataType dataType);

    bool isSystemBufferType() const { return ::mlir::tt::ttnn::isSystemBufferType(getBufferType()); }
    bool isDeviceBufferType() const { return ::mlir::tt::ttnn::isDeviceBufferType(getBufferType()); }
    bool isTiled() const;
    bool hasShardedTensorMemoryLayout() const;
    bool hasShardedL1TensorMemoryLayout() const;
    bool hasInterleavedL1TensorMemoryLayout() const;
    bool hasInterleavedDRAMTensorMemoryLayout() const;
    bool hasDRAMBufferType() const;
    bool hasL1BufferType() const;
    Layout getLayout() const;
    std::optional<TensorMemoryLayout> getMemLayoutOpt() const;
    Type getElementType() const;
    Type getScalarElementType() const;
    uint64_t getShardSizeInBytes() const;
    BufferType getBufferType() const;
    ttcore::DataType getDataType() const;
    uint64_t getElementSizeBytes() const;
    static llvm::SmallVector<int64_t> calculateLogicalShardShapeForSharding(ArrayRef<int64_t> tensorShape, mlir::AffineMap linear, ttcore::GridAttr grid);
    static llvm::SmallVector<int64_t> calculateLogicalShardShapeForL1Interleaved(ArrayRef<int64_t> tensorShape, Type elementType, mlir::AffineMap linear, ttcore::GridAttr grid);
    llvm::SmallVector<int64_t> getShardShape() const;
    llvm::SmallVector<int64_t> getScalarShardShape() const;
    llvm::SmallVector<int64_t> getTiledShape(ArrayRef<int64_t> logicalTensorShape) const;
    std::pair<std::int64_t, std::int64_t> getDefaultCollapseIntervals() const;
  }];

  let genVerifyDecl = 1;
}

def TTNN_KernelSemaphoreAttr: TTNN_Attr<"KernelSemaphore", "kernel_semaphore"> {
  let summary = "Kernel semaphore descriptor";
  let description = [{
    Kernel semaphore descriptor for TTNN generic operation.
  }];

  let parameters = (ins "KernelCoreType":$core_type,
                        TTNN_CoreRangeSetAttr:$core_ranges,
                        "uint32_t":$initial_value);
  let assemblyFormat = "`<` struct(params) `>`";

  let genVerifyDecl = 1;
}

def TTNN_KernelCBFormatAttr: TTNN_Attr<"KernelCBFormat", "kernel_cb_format"> {
  let summary = "Kernel CB format descriptor";
  let description = [{
    Kernel CB format descriptor for TTNN generic operation.
  }];

  let parameters = (ins "uint32_t":$buffer_index,
                        "ttcore::DataType":$dtype,
                        "uint32_t":$page_size);
  let assemblyFormat = "`<` struct(params) `>`";

  let genVerifyDecl = 1;
}

def TTNN_KernelCBAttr: TTNN_Attr<"KernelCB", "kernel_cb"> {
  let summary = "Kernel CB descriptor";
  let description = [{
    Kernel CB descriptor for TTNN generic operation.
  }];

  let parameters = (ins "uint32_t":$total_size,
                        TTNN_CoreRangeSetAttr:$core_ranges,
                        ArrayRefParameter<"KernelCBFormatAttr">:$formats);
  let assemblyFormat = "`<` struct($total_size, $core_ranges) `,` `formats` `=` `[` $formats `]` `>`";

  let genVerifyDecl = 1;
}

def TTNN_KernelArgCBBufferIndexAttr : TTNN_Attr<"KernelArgCBBufferIndex", "kernel_arg_cb_buffer_index"> {
  let summary = "CB argument";
  let description = [{
    TTNN generic kernel argument for CB buffer index.
  }];

  let parameters = (ins "size_t":$buffer_index);
  let assemblyFormat = "`<` $buffer_index `>`";
}

def TTNN_KernelArgAddressOfTensorAttr : TTNN_Attr<"KernelArgAddressOfTensor", "kernel_arg_address_of_tensor"> {
  let summary = "Address of tensor argument";
  let description = [{
    TTNN generic kernel argument for address of tensor at postion in operation tensors array.
  }];

  let parameters = (ins "size_t":$tensor_index);
  let assemblyFormat = "`<` $tensor_index `>`";
}

def TTNN_KernelArgSemaphoreAtAttr : TTNN_Attr<"KernelArgSemaphoreAt", "kernel_arg_semaphore_at"> {
  let summary = "Semaphore argument";
  let description = [{
    TTNN generic kernel argument for semaphore at position in semaphores array in program.
  }];

  let parameters = (ins "size_t":$semaphore_index);
  let assemblyFormat = "`<` $semaphore_index `>`";
}

def TTNN_ComputeKernelAttr: TTNN_Attr<"ComputeKernel", "compute_kernel", [TTNN_KernelInterface]> {
  let summary = "Compute kernel";
  let description = [{
    Compute kernel descriptor for TTNN generic operation.
  }];

  let parameters = (ins "SymbolRefAttr":$symbol_ref,
                        TTNN_CoreRangeSetAttr:$core_ranges,
                        "ComputeKernelMathFidelity":$math_fidelity,
                        "bool":$fp32_dest_acc_en,
                        "bool":$dst_full_sync_en,
                        ArrayRefParameter<"ComputeKernelUnpackToDestMode">:$unpack_to_dest_modes,
                        "bool":$bfp8_pack_precise,
                        "bool":$math_approx_mode,
                        OptionalArrayRefParameter<"mlir::Attribute">:$common_rt_args,
                        OptionalArrayRefParameter<"mlir::Attribute">:$ct_args
                        );
  let assemblyFormat = "`<` struct($symbol_ref, $core_ranges, $math_fidelity, $fp32_dest_acc_en, $dst_full_sync_en)  `,` `unpack_to_dest_modes` `=` `[` $unpack_to_dest_modes `]` `,` struct($bfp8_pack_precise, $math_approx_mode) `,` `ct_args` `=` `[` (`]`) : ($ct_args^ `]`)?  `,` `common_rt_args` `=` `[` (`]`) : ($common_rt_args^ `]`)? `>`";

  let genVerifyDecl = 1;
}

def TTNN_ReadKernelAttr: TTNN_Attr<"ReadKernel", "read_kernel", [TTNN_KernelInterface]> {
  let summary = "Read kernel";
  let description = [{
    Read kernel descriptor for TTNN generic operation.
  }];

  let parameters = (ins "SymbolRefAttr":$symbol_ref,
                        TTNN_CoreRangeSetAttr:$core_ranges,
                        OptionalArrayRefParameter<"mlir::Attribute">:$common_rt_args,
                        OptionalArrayRefParameter<"mlir::Attribute">:$ct_args);
  let assemblyFormat = "`<` struct($symbol_ref, $core_ranges) `,` `ct_args` `=` `[` (`]`) : ($ct_args^ `]`)?  `,` `common_rt_args` `=` `[` (`]`) : ($common_rt_args^ `]`)? `>`";

  let genVerifyDecl = 1;
}

def TTNN_WriteKernelAttr: TTNN_Attr<"WriteKernel", "write_kernel", [TTNN_KernelInterface]> {
  let summary = "Write kernel";
  let description = [{
    Write kernel descriptor for TTNN generic operation.
  }];

  let parameters = (ins "SymbolRefAttr":$symbol_ref,
                        TTNN_CoreRangeSetAttr:$core_ranges,
                        OptionalArrayRefParameter<"mlir::Attribute">:$common_rt_args,
                        OptionalArrayRefParameter<"mlir::Attribute">:$ct_args);
  let assemblyFormat = "`<` struct($symbol_ref, $core_ranges) `,` `ct_args` `=` `[` (`]`) : ($ct_args^ `]`)?  `,` `common_rt_args` `=` `[` (`]`) : ($common_rt_args^ `]`)? `>`";

  let genVerifyDecl = 1;
}

def TTNN_ProgramAttr: TTNN_Attr<"Program", "program"> {
  let summary = "Program";
  let description = [{
    Program descriptor for TTNN generic operation.
  }];

  let parameters = (ins ArrayRefParameter<"mlir::Attribute">:$kernels,
                        OptionalArrayRefParameter<"KernelCBAttr">:$cbs,
                        OptionalArrayRefParameter<"KernelSemaphoreAttr">:$semaphores);
  let assemblyFormat = "`<` `kernels` `=` `[` $kernels `]` `,` `cbs` `=` `[` (`]`) : ($cbs^ `]`)? `,` `semaphores` `=` `[` (`]`) : ($semaphores^ `]`)? `>`";

  let genVerifyDecl = 1;
}

def TTNN_TraceIdAttr : TTNN_Attr<"TraceId", "trace_id"> {
  let summary = "TTNN Trace ID attribute";
  let description = [{
    A unit attribute indicating that the tensor represents `trace_id`.
  }];
}

#endif  // TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSATTRS_TD
