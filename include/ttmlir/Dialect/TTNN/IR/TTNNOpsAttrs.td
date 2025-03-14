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

def TTNN_CoreRangeAttr : TTNN_Attr<"CoreRange", "core_range"> {
  let summary = "TTNN grid attribute";
  let description = [{
    TTNN grid attribute
  }];

  let parameters = (ins ArrayRefParameter<"int64_t">:$offset,
                        ArrayRefParameter<"int64_t">:$size);
  let assemblyFormat = "`<` custom<DimensionList>($offset) `,` custom<DimensionList>($size) `>`";

  let extraClassDeclaration = [{
      static CoreRangeAttr get(::mlir::MLIRContext *context, ::mlir::tt::GridAttr grid, SmallVector<int64_t> offset = {0, 0})
      {
        assert(grid.getShape().size() == 2 && "Grid shape must be 2D for now");
        return CoreRangeAttr::get(context, {0, 0}, grid.getShape());
      }
  }];
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

def TTNN_Conv2dConfigAttr : TTNN_Attr<"Conv2dConfig", "conv2d_config"> {
  let summary = "TTNN Conv2dConfig attribute";
  let description = [{
    TTNN conv2d config attribute
  }];

  let parameters = (ins
                        "DataType":$dtype,
                        "DataType":$weightsDtype,
                        "StringAttr":$activation,
                        "IntegerAttr":$inputChannelsAlignment,
                        "BoolAttr":$deallocateActivation,
                        "BoolAttr":$reallocateHaloOutput,
                        "IntegerAttr":$actBlockHOverride,
                        "IntegerAttr":$actBlockWDiv,
                        "BoolAttr":$reshardIfNotOptimal,
                        "BoolAttr":$overrideShardingConfig,
                        OptionalParameter<"TensorMemoryLayoutAttr", "TTNN tensor memory layout">:$shardLayout,
                        // TODO: Finish adding this attribute https://github.com/tenstorrent/tt-mlir/issues/2130
                        OptionalParameter<"Attribute", "TTNN core grid">:$coreGrid,
                        "BoolAttr":$transposeShards,
                        "Layout":$outputLayout,
                        "BoolAttr":$enableActDoubleBuffer,
                        "BoolAttr":$enableWeightsDoubleBuffer,
                        "BoolAttr":$enableSplitReader,
                        "BoolAttr":$enableSubblockPadding);

let assemblyFormat = "`<` `dtype` `=` $dtype `,` "
                     "`weightsDtype` `=` $weightsDtype `,` "
                     "`activation` `=` $activation `,` "
                     "`inputChannelsAlignment` `=` $inputChannelsAlignment `,` "
                     "`deallocateActivation` `=` $deallocateActivation `,` "
                     "`reallocateHaloOutput` `=` $reallocateHaloOutput `,` "
                     "`actBlockHOverride` `=` $actBlockHOverride `,` "
                     "`actBlockWDiv` `=` $actBlockWDiv `,` "
                     "`reshardIfNotOptimal` `=` $reshardIfNotOptimal `,` "
                     "`overrideShardingConfig` `=` $overrideShardingConfig `,` "
                     "(`shardLayout` `=` $shardLayout^ `,`)? "
                     "(`coreGrid` `=` $coreGrid^ `,`)? "
                     "`transposeShards` `=` $transposeShards `,` "
                     "`outputLayout` `=` $outputLayout `,` "
                     "`enableActDoubleBuffer` `=` $enableActDoubleBuffer `,` "
                     "`enableWeightsDoubleBuffer` `=` $enableWeightsDoubleBuffer `,` "
                     "`enableSplitReader` `=` $enableSplitReader `,` "
                     "`enableSubblockPadding` `=` $enableSubblockPadding `>`";
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
