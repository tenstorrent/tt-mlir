// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSATTRS_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "ttmlir/Dialect/TTNN/IR/TTNNBase.td"
include "ttmlir/Dialect/TTNN/IR/TTNNOpsEnums.td"

//===----------------------------------------------------------------------===//
// TTNN attr definitions
//===----------------------------------------------------------------------===//

class TTNN_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Attribute">
    : AttrDef<TTNN_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
  let attrName = "ttnn." # attrMnemonic;
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

def TTNN_ShardSpecAttr : TTNN_Attr<"ShardSpec", "shard_spec"> {
  let summary = "TTNN ShardSpec attribute";
  let description = [{
    TTNN ShardSpec attribute
  }];

  // TODO (#620): Add other fields like core_ranges, shard orientation etc.
  let parameters = (ins AttrParameter<"ShapeAttr", "">:$shardShape);
  let assemblyFormat = "`<` params `>`";
}

def TTNN_MemoryConfigAttr : TTNN_Attr<"MemoryConfig", "memory_config"> {
  let summary = "TTNN MemoryConfig attribute";
  let description = [{
    TTNN memory config attribute
  }];

  let parameters = (ins AttrParameter<"BufferTypeAttr", "">:$bufferType,
                        AttrParameter<"ShardSpecAttr", "">:$shardSpec,
                        OptionalParameter<"TensorMemoryLayoutAttr">:$tensorMemoryLayout);

  let assemblyFormat = "`<` params `>`";

  let extraClassDeclaration = [{
    llvm::ArrayRef<int64_t> getShardShape(bool convertTileToScalar = true) const;
    MemoryConfigAttr withBufferType(::mlir::MLIRContext *context, BufferType bufferType);
    MemoryConfigAttr withMemoryLayout(::mlir::MLIRContext *context, TensorMemoryLayout memLayout);
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
      num_global_cb_receivers = 0
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
                        "uint64_t":$num_global_cb_receivers);

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
      $num_global_cb_receivers
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
    TTNN conv2d config attribute
  }];

  let parameters = (ins "DataType":$dtype,
                        "DataType":$weights_dtype,
                        "StringAttr":$activation,
                        "uint32_t":$input_channels_alignment,
                        "bool":$deallocate_activation,
                        "bool":$reallocate_halo_output,
                        "uint32_t":$act_block_h_override,
                        "uint32_t":$act_block_w_div,
                        "bool":$reshard_if_not_optimal,
                        "bool":$override_sharding_config,
                        OptionalParameter<"TensorMemoryLayoutAttr">:$shard_layout,
                        OptionalParameter<"CoreRangeSetAttr">:$core_grid,
                        "bool":$transpose_shards,
                        "Layout":$output_layout,
                        "bool":$preprocess_weights_on_device,
                        "bool":$always_preprocess_weights,
                        "bool":$enable_act_double_buffer,
                        "bool":$enable_weights_double_buffer,
                        "bool":$enable_split_reader,
                        "bool":$enable_subblock_padding,
                        "bool":$in_place);

  let assemblyFormat = [{
    `<` struct(
      qualified($dtype),
      qualified($weights_dtype),
      $activation,
      $input_channels_alignment,
      $deallocate_activation,
      $reallocate_halo_output,
      $act_block_h_override,
      $act_block_w_div,
      $reshard_if_not_optimal,
      $override_sharding_config,
      qualified($shard_layout),
      qualified($core_grid),
      $transpose_shards,
      qualified($output_layout),
      $preprocess_weights_on_device,
      $always_preprocess_weights,
      $enable_act_double_buffer,
      $enable_weights_double_buffer,
      $enable_split_reader,
      $enable_subblock_padding,
      $in_place
    ) `>`
  }];
}

def TTNN_MeshShapeAttr : TTNN_Attr<"MeshShape", "mesh_shape"> {
  let summary = "TTNN Mesh Shape";
  let description = [{
    TTNN mesh shape
  }];

  let parameters = (ins "int64_t":$y, "int64_t":$x);
  let assemblyFormat = "custom<VargDimensionList>($y, $x)";
}

def TTNN_TTNNLayoutAttr: TTNN_Attr<"TTNNLayout", "ttnn_layout"> {
  let summary = "Tensor encoding attribute used for types in ttnn";
  let description = [{
    Layout attribute in ttnn. This attribute is used to encode different information about tensor memory layout.
    Here is how tensor will look like after layout tensor<32x32x64xf32, #ttnn.ttnn_layout<linear, grid, memref, mem_layout>>
    Lets break down what each parameter means:
    - linear: An affine map that defines how the logical tensor dimensions map to physical space.
    - grid: The grid shape (of tensix cores) where tensor is divided onto.
    - memref: A memref is used to describe shard size and memory space. Shard size is calculated by dividing the tensor size by grid size.
    - mem_layout: The layout of the tensor in memory. For tensor on host it should be None. For tensor on device
    it can be interleaved or sharded.
    - mesh_sharding: The mesh of the tensor in multi-devices.
  }];

  let parameters = (ins AttrParameter<"AffineMap", "An affine map that defines how the logical tensor dimensions map to a grid shape.">:$linear,
                        AttrParameter<"GridAttr", "The grid shape that this tensor is divided onto.">:$grid,
                        AttrParameter<"MemRefType", "A memref that describes the physical footprint allocation of the shard. It must also have a shape with rank equal to grid.">:$memref,
                        OptionalParameter<"TensorMemoryLayoutAttr", "TTNN tensor memory layout">:$mem_layout,
                        OptionalParameter<"TensorMeshShardingAttr", "TT mesh sharding attr">:$tensor_mesh_sharding);
  let assemblyFormat = "`<` $linear`,` $grid`,` (`mesh` `=` $tensor_mesh_sharding^ `,`)? $memref (`,` $mem_layout^)? `>`";
  let extraClassDeclaration = [{
    static TTNNLayoutAttr get(::mlir::MLIRContext *context,
                        ArrayRef<int64_t> tensorShape,
                        Type elementType,
                        BufferType bufferType,
                        GridAttr grid,
                        TensorMemoryLayoutAttr memoryLayoutAttr = nullptr,
                        TensorMeshShardingAttr tensorMeshShardingAttr = nullptr,
                        ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals = {{0, -1}});

    TTNNLayoutAttr withGrid(::mlir::MLIRContext *context, ArrayRef<int64_t> tensorShape, GridAttr grid, ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals = {{0, -1}});
    TTNNLayoutAttr withGrid(::mlir::MLIRContext *context,
                        RankedTensorType ty,
                        GridAttr grid,
                        ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals = {{0, -1}});
    TTNNLayoutAttr withElementType(::mlir::MLIRContext *context, Type elementType, ArrayRef<int64_t> tensorShape, ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals = {{0, -1}});
    TTNNLayoutAttr withBufferType(::mlir::MLIRContext *context, BufferType bufferType);
    TTNNLayoutAttr withMemoryLayout(::mlir::MLIRContext *context, TensorMemoryLayoutAttr memLayoutAttr);
    TTNNLayoutAttr withMemoryLayout(::mlir::MLIRContext *context, TensorMemoryLayout memLayout);
    TTNNLayoutAttr withShardShape(::mlir::MLIRContext *context, llvm::SmallVector<int64_t> shardShape);
    TTNNLayoutAttr withTensorShape(::mlir::MLIRContext *context, ArrayRef<int64_t> tensorShape);

    bool isSystemBufferType() const { return ::mlir::tt::ttnn::isSystemBufferType(getBufferType()); }
    bool isDeviceBufferType() const { return ::mlir::tt::ttnn::isDeviceBufferType(getBufferType()); }
    bool isMeshDeviceTensor() const { return ::mlir::tt::ttnn::isMeshDeviceTensor(getTensorMeshSharding()); }
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
    DataType getDataType() const;
    uint64_t getElementSizeBytes() const;
    static llvm::SmallVector<int64_t> calculateLogicalShardShapeForSharding(ArrayRef<int64_t> tensorShape, mlir::AffineMap linear, GridAttr grid);
    static llvm::SmallVector<int64_t> calculateLogicalShardShapeForL1Interleaved(ArrayRef<int64_t> tensorShape, Type elementType, mlir::AffineMap linear, GridAttr grid);
    llvm::SmallVector<int64_t> getShardShape() const;
    llvm::SmallVector<int64_t> getScalarShardShape() const;
    AffineMap getIdentityTileLinearMap() const;
    llvm::SmallVector<int64_t> getTiledShape(ArrayRef<int64_t> logicalTensorShape) const;
    AffineMap replaceMemoryMapSymbolsWithShardShape(AffineMap physicalMemoryMap) const;
  }];

  let genVerifyDecl = 1;
}

#endif  // TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSATTRS_TD
