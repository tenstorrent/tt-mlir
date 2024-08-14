// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_TTOPSTYPES_TD
#define TTMLIR_TTMLIR_TTOPSTYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "ttmlir/Dialect/TT/IR/TTBase.td"
include "ttmlir/Dialect/TT/IR/TTOpsEnums.td"

//===----------------------------------------------------------------------===//
// TT attr definitions
//===----------------------------------------------------------------------===//
// Should Attr be a separate file?

class TT_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Attribute">
    : AttrDef<TT_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
  let attrName = "tt." # attrMnemonic;
}

def TT_GridAttr : TT_Attr<"Grid", "grid"> {
  let summary = "TT grid attribute";
  let description = [{
    TT grid attribute
  }];

  let parameters = (ins ArrayRefParameter<"int64_t">:$shape,
                        DefaultValuedParameter<
                          "AffineMap",
                          "$_builder.getEmptyAffineMap()">:$mapping);
  let assemblyFormat = "`<` custom<DimensionList>($shape) (`,` $mapping^)? `>`";

  let extraClassDeclaration = [{
      static GridAttr get(::mlir::MLIRContext *context) {
        return GridAttr::get(context, {1, 1});
      }

      static GridAttr get(::mlir::MLIRContext *context, ArrayRef<int64_t> shape) {
        return GridAttr::get(context, shape, AffineMap::get(context));
      }
  }];
}

def TT_ArchAttr : EnumAttr<TT_Dialect, TT_Arch, "arch"> {
  let assemblyFormat = "`<` $value `>`";
}

def TT_CoreCoordAttr : TT_Attr<"CoreCoord", "core_coord"> {
  let summary = "TT core_coord attribute";
  let description = [{
    TT core_coord attribute containing a single physical core coordinate.
  }];

  let parameters = (ins "int64_t":$y, "int64_t":$x);
  let assemblyFormat = "`(` $y `,` $x `)`";
}


def TT_ChipPhysicalCoresAttr : TT_Attr<"ChipPhysicalCores", "chip_physical_cores"> {
  let summary = "TT chip_physical_cores attribute";
  let description = [{
    TT chip_physical_cores attribute containing arrays of physical cores by core type in order of logical cores.
  }];

  let parameters = (ins ArrayRefParameter<"CoreCoordAttr">:$worker, ArrayRefParameter<"CoreCoordAttr">:$dram, OptionalArrayRefParameter<"CoreCoordAttr">:$eth, OptionalArrayRefParameter<"CoreCoordAttr">:$eth_inactive);
  let assemblyFormat = "`{` `worker` `=` `[` $worker `]` `dram` `=` `[` $dram `]` (`eth` `=` `[` $eth^ `]`)? (`eth_inactive` `=` `[` $eth_inactive^ `]`)? `}`";
}

def TT_ChipDescAttr : TT_Attr<"ChipDesc", "chip_desc"> {
  let summary = "TT chip_desc attribute";
  let description = [{
    TT chip_desc attribute
  }];

  let parameters = (ins "ArchAttr":$arch,
                    ArrayRefParameter<"int64_t">:$grid,
                    "unsigned":$l1Size,
                    "unsigned":$numDramChannels,
                    "unsigned":$dramChannelSize,
                    "unsigned":$nocL1AddressAlignBytes,
                    "unsigned":$pcieAddressAlignBytes,
                    "unsigned":$nocDRAMAddressAlignBytes,
                    "unsigned":$l1UnreservedBase,
                    "unsigned":$eriscL1UnreservedBase,
                    "unsigned":$dramUnreservedBase,
                    "ChipPhysicalCoresAttr":$chipPhysicalCores);
  let assemblyFormat = [{`{` `arch` `=` $arch `,`
                             `grid` `=` custom<DimensionList>($grid) `,`
                             `l1_size` `=` $l1Size `,`
                             `num_dram_channels` `=` $numDramChannels `,`
                             `dram_channel_size` `=` $dramChannelSize `,`
                             `noc_l1_address_align_bytes` `=` $nocL1AddressAlignBytes `,`
                             `pcie_address_align_bytes` `=` $pcieAddressAlignBytes `,`
                             `noc_dram_address_align_bytes` `=` $nocDRAMAddressAlignBytes  `,`
                             `l1_unreserved_base` `=` $l1UnreservedBase `,`
                             `erisc_l1_unreserved_base` `=` $eriscL1UnreservedBase `,`
                             `dram_unreserved_base` `=` $dramUnreservedBase `,`
                             `physical_cores` `=` $chipPhysicalCores `}`}];

  let extraClassDeclaration = [{
    unsigned getUsableL1Size() const { return getL1Size() - getL1UnreservedBase(); }
    unsigned getUsableDramChannelSize() const { return getDramChannelSize() - getDramUnreservedBase(); }
  }];
}

def TT_ChipCoordAttr : TT_Attr<"ChipCoord", "chip_coord"> {
  let summary = "TT chip_coord attribute";
  let description = [{
    TT chip_coord attribute
  }];

  let parameters = (ins "unsigned":$rack, "unsigned":$shelf, "unsigned":$y, "unsigned":$x);
  let assemblyFormat = "`<` $rack `,` $shelf `,` $y `,` $x `>`";
}

def TT_ChipChannelAttr : TT_Attr<"ChipChannel", "chip_channel"> {
  let summary = "TT chip_channel attribute";
  let description = [{
    TT chip_channel attribute
  }];

  let parameters = (ins "unsigned":$deviceId0,
                        ArrayRefParameter<"int64_t">:$ethernetCoreCoord0,
                        "unsigned":$deviceId1,
                        ArrayRefParameter<"int64_t">:$ethernetCoreCoord1);
  let assemblyFormat = "`<` `[` $deviceId0 `,` $ethernetCoreCoord0 `]` `,` `[` $deviceId1 `,` $ethernetCoreCoord1 `]` `>`";
}

def TT_SystemDescAttr : TT_Attr<"SystemDesc", "system_desc"> {
  let summary = "TT system_desc attribute";
  let description = [{
    TT system_desc attribute
  }];

  let parameters = (ins ArrayRefParameter<"ChipDescAttr">:$chipDescs,
                        ArrayRefParameter<"unsigned">:$chipDescIndices,
                        ArrayRefParameter<"ChipCapabilityAttr">:$chipCapabilities,
                        ArrayRefParameter<"ChipCoordAttr">:$chipCoords,
                        OptionalArrayRefParameter<"ChipChannelAttr">:$chipChannels);
  let assemblyFormat = "`<` `[` $chipDescs `]` `,` `[` $chipDescIndices `]` `,` `[` $chipCapabilities `]` `,` `[` $chipCoords `]` (`,` `[` $chipChannels^ `]`)? `>`";

  let extraClassDeclaration = [{
    static tt::SystemDescAttr getDefault(MLIRContext *context);
    static tt::SystemDescAttr getFromPath(MLIRContext *context, std::string& path);
    unsigned getAddressAlignBytes(unsigned chipIndex = 0) const;
    unsigned getAddressAlignBytes(MemorySpace memorySpace, unsigned chipIndex = 0) const;
    unsigned getNocL1AddressAlignBytes(unsigned chipIndex = 0) const;
    unsigned getNocDRAMAddressAlignBytes(unsigned chipIndex = 0) const;
    unsigned getPcieAddressAlignBytes(unsigned chipIndex = 0) const;
  }];
}

def TT_LayoutAttr : TT_Attr<"Layout", "layout"> {
  let summary = "Tensor layout attribute";
  let description = [{
    The tensor layout attribute captures how tensor data is sharded across a grid of devices, cores, and
    is laid out in memory.

    Some high level goals
      - **Logical shapes**: Keep the original tensor shape and rank intact and agnostic
        to underlying storage layout.
        Keeping the logical shapes not only makes some graph transformations vastly
        simpler, in particular convs, but it makes the lowered IR much easier to read
        and reason about.  The original tensor shapes leave breadcrumbs that make it
        much easier to map back to the input representation.
      - **Flexible sharding**: Enable flexibility in choosing grid shape, to get better
        parallelization and avoid resharding. This is particularly important in cases
        where tensor shapes are not clean powers of two and would otherwise force our
        hand in choosing non-optimal grid shapes.
      - **Logical-Physical Isomorphism**: Encode this information with just a few
        attributes to enable derived conversions from logical to physical layout and back.
      - **Explicit**: A single source of truth.
      - Enable a direct way to query padded regions.

    Please refer to the [Tensor Layout Spec](https://tenstorrent.github.io/tt-mlir/specs/tensor-layout.html) for more in depth documentation.

    Examples:
    ```mlir
    tensor<8x300xf32,
      #tt.layout<(d0, d1) -> (d0, d1),
        undef,
        <1x2>,
        memref<8x150xf32, #tt.memory_space<l1>>
      >
    >

    tensor<8x96x32xf32,
      #tt.layout<(d0, d1, d2) -> (d0 * 96 + d1, d2),
        undef,
        <2x1>,
        memref<384x32xf32, #tt.memory_space<l1>>
      >
    >

    tensor<8x96x32xf32,
      #tt.layout<(d0, d1, d2) -> (d0 * 96 + d1, d1, d2),
        undef,
        <2x1x2>,
        memref<384x96x16xf32, #tt.memory_space<l1>>
      >
    >

    tensor<5x3x2x2x7x32x32xf32,
      #tt.layout<
        (d0, d1, d2, d3, d4, d5, d6)
          -> (d0 * 2688 + d1 * 896 + d2 * 448 + d3 * 224 + d4 * 32 + d5, d4, d5, d6),
        undef,
        <3x2x2x2>,
        memref<4480x4x16x16xf32, #tt.memory_space<l1>>
      >
    >
    ```
  }];

  let parameters = (ins AttrParameter<"AffineMap", "An affine map that defines how the logical tensor dimensions map to a grid shape.">:$linear,
                        AttrParameter<"OOBVal", "A tracked out of bounds value that fills padding space.">:$oob_val,
                        AttrParameter<"GridAttr", "The grid shape that this tensor is divided onto.">:$grid,
                        AttrParameter<"MemRefType", "A memref that describes the physical footprint allocation of the shard. It must also have a shape with rank equal to grid.">:$memref);
  let assemblyFormat = "`<` $linear`,` $oob_val`,` $grid`,` $memref `>`";

  let extraClassDeclaration = [{
      static LayoutAttr get(::mlir::MLIRContext *context,
                            ArrayRef<int64_t> tensorShape,
                            Type elementType,
                            MemorySpace memorySpace = MemorySpace::System,
                            GridAttr grid = {},
                            ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals = {{0, -1}},
                            OOBVal oobVal = OOBVal::Undef);
      static LayoutAttr get(::mlir::MLIRContext *context,
                            RankedTensorType ty,
                            MemorySpace memorySpace = MemorySpace::System,
                            GridAttr grid = {},
                            ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals = {{0, -1}},
                            OOBVal oobVal = OOBVal::Undef);
      static LayoutAttr get(::mlir::MLIRContext *context,
                            RankedTensorType ty,
                            MemorySpace memorySpace,
                            Type elementType);
      LayoutAttr withGrid(::mlir::MLIRContext *context, ArrayRef<int64_t> tensorShape, GridAttr grid, ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals = {{0, -1}});
      LayoutAttr withGrid(::mlir::MLIRContext *context,
                          RankedTensorType ty,
                          GridAttr grid,
                          ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals = {{0, -1}});
      LayoutAttr withElementType(::mlir::MLIRContext *context, Type elementType);
      LayoutAttr withMemorySpace(::mlir::MLIRContext *context, MemorySpace memorySpace);

      MemorySpace getMemorySpace() const;
      bool isSystemMemorySpace() const { return ::mlir::tt::isSystemMemorySpace(getMemorySpace()); }
      bool isDeviceMemorySpace() const { return ::mlir::tt::isDeviceMemorySpace(getMemorySpace()); }
      bool isTiled() const;
      Type getElementType() const;
      Type getScalarElementType() const;
      uint64_t getElementSizeBytes() const;
      llvm::SmallVector<int64_t> getStride(ArrayRef<int64_t> logicalShape) const;
      llvm::SmallVector<int64_t> getPhysicalShape(ArrayRef<int64_t> logicalShape) const;
      llvm::SmallVector<int64_t> getShardShape() const;
      AffineMap projectOnto(ArrayRef<int64_t> logicalTensorShape, GridAttr grid) const;
  }];
}

def TT_BufferAccessAttr : EnumAttr<TT_Dialect, TT_BufferAccess, "buffer_access"> {
  let assemblyFormat = "$value";
}

def TT_BufferAttr : TT_Attr<"Buffer", "buffer", []> {
  let summary = "Buffer attribute in TT dialect";
  let description = [{
    Describes the physical footprint and layout of a buffer in L1. Its memref must also have a shape with rank equal to DeviceAttr grid.
    It also carries a buffer access attribute which can be one of:
    - Alias: This buffer aliases a persistent Tensor L1 allocation directly. Implies that no datamovement occurs and the compute kernel
             just accesses the local allocation directly.
    - Stream: This buffer is a temporary destination as a means to get remote data for local computation.  Remote data is most likely a
              a tensor that is allocated in dram, but could also be data from a remote core.
  }];
  let parameters = (ins AttrParameter<"MemRefType", "A memref that describes the physical footprint and layout of the buffer. It must also have a shape with rank equal to DeviceAttr grid.">:$memref,
                        AttrParameter<"BufferAccess", "How data is accessed through this buffer, alias or stream.">:$buffer_access);
  let assemblyFormat = "`<` $memref `,` $buffer_access `>`";

  let extraClassDeclaration = [{
      ::mlir::Type getElementType() const;
      ::llvm::SmallVector<int64_t> getShape() const;
  }];
}

def TT_DeviceAttr : TT_Attr<"Device", "device", []> {
  let summary = "Device attribute in TT dialect";
  let description = [{
  }];
  let parameters = (ins TT_GridAttr:$grid, ArrayRefParameter<"unsigned">:$chipIds);
  let assemblyFormat = "`<` qualified($grid) `,` `[` $chipIds `]` `>`";

  let extraClassDeclaration = [{
      static DeviceAttr get(::mlir::MLIRContext *context, ArrayRef<int64_t> shape, AffineMap physicalGridMapping, ArrayRef<unsigned> chipIds) {
        return DeviceAttr::get(context, GridAttr::get(context, shape, physicalGridMapping), chipIds);
      }
      static DeviceAttr get(::mlir::MLIRContext *context, SystemDescAttr systemDesc, ArrayRef<unsigned> chipIds);
      static DeviceAttr get(::mlir::MLIRContext *context, SystemDescAttr systemDesc, bool enableMultichip = false);
  }];

  let genVerifyDecl = 1;
}

def TT_MemorySpaceAttr : EnumAttr<TT_Dialect, TT_MemorySpace, "memory_space"> {
  let assemblyFormat = "`<` $value `>`";
}

def TT_OOBValAttr : EnumAttr<TT_Dialect, TT_OOBVal, "oob_val"> {
  let assemblyFormat = "`<` $value `>`";
}

def TT_IteratorTypeAttr : EnumAttr<TT_Dialect, TT_IteratorType, "iterator_type"> {
  let assemblyFormat = "`<` $value `>`";
}

def TT_IteratorTypeArrayAttr : TypedArrayAttrBase<TT_IteratorTypeAttr, "">;

def TT_OperandConstraintAttr : EnumAttr<TT_Dialect, TT_OperandConstraint, "operand_constraint"> {
  let assemblyFormat = "`<` $value `>`";
}

def TT_OperandConstraintArrayAttr : TypedArrayAttrBase<TT_OperandConstraintAttr, "">;

def TT_ArgumentAllocationAttr : TT_Attr<"ArgumentAllocation", "arg_alloc", []> {
  let summary = "Argument allocation attribute in TT dialect";
  let description = [{
    Holds the metadata for the allocation of an function argument i.e. for graph inputs.
  }];
  let parameters = (ins "uint64_t":$address, "uint64_t":$size, "MemorySpace":$memorySpace);
  let assemblyFormat = "`<` $address `,` $size `,` $memorySpace `>`";
}

def TT_CoreRangeAttr : TT_Attr<"CoreRange", "core_range"> {
  let summary = "Attribute representing range of core coordinates.";
  let description = [{
    Attribute representing range of core coordinates.
  }];

  let parameters = (ins ArrayRefParameter<"int64_t">:$offset,
                        ArrayRefParameter<"int64_t">:$size);

  let assemblyFormat = "`<` custom<DimensionList>($offset) `,` custom<DimensionList>($size) `>`";

  let extraClassDeclaration = [{
      static CoreRangeAttr get(::mlir::MLIRContext *context, ::mlir::tt::GridAttr grid, SmallVector<int64_t> offset = {0, 0})
      {
        assert(grid.getShape().size() == 2 && "Grid shape must be 2D for now");
        return CoreRangeAttr::get(context, offset, grid.getShape());
      }
  }];
}

// An array of core range attributes.
def TT_CoreRangeArrayAttr : TypedArrayAttrBase<TT_CoreRangeAttr, "">;

def TT_CircularBufferAttributesAttr : TT_Attr<"CircularBufferAttributes", "circular_buffer_attributes"> {
  let summary = "Circular buffer attributes.";

  let description = [{
    Attribute representing circular_buffer_attributes_t structure.
  }];

  let parameters = (ins 
    "CB":$cb_id,
    TT_CoreRangeAttr:$core_range,
    "uint32_t":$total_size,
    "uint32_t":$page_size,
    "DataType":$data_format
  );

  let assemblyFormat = "`<` $cb_id `,` $core_range `,` $total_size `,` $page_size `,` $data_format `>`";
}

// An array of circular buffer attributes.
def TT_CircularBufferAttributesArrayAttr : TypedArrayAttrBase<TT_CircularBufferAttributesAttr, "">;

def TT_DataMovementConfigAttr : TT_Attr<"DataMovementConfig", "data_movement_config"> {
  let summary = "Data movement config.";
  let description = [{
    Attribute representing DataMovementConfig structure.
  }];

  let parameters = (ins 
    "DataMovementType":$data_movement_type,
    ArrayRefParameter<"uint32_t">:$compile_args);

  let assemblyFormat = "`<` $data_movement_type `,` `<` $compile_args `>` `>`";
}

def TT_DataMovementAttributesAttr : TT_Attr<"DataMovementAttributes", "data_movement_attributes"> {
  let summary = "Data movement attributes.";
  let description = [{
    Attribute representing the data_movement_attributes_t structure.
  }];

  let parameters = (ins 
    TT_CoreRangeAttr:$core_range,
    "::mlir::StringAttr":$kernel_path,
    TT_DataMovementConfigAttr:$data_movement_config);
    // TODO runtime_args missing

  let assemblyFormat = "`<` $core_range `,` $kernel_path `,` $data_movement_config `>`";
}

// An array of data movement attributes.
def TT_DataMovementAttributesArrayAttr : TypedArrayAttrBase<TT_DataMovementAttributesAttr, "">;

def TT_ComputeConfigAttr : TT_Attr<"ComputeConfig", "compute_config"> {
  let summary = "Compute config.";
  let description = [{
    Attribute representing ComputeConfig structure.
  }];

  let parameters = (ins 
    "MathFidelity":$math_fidelity,
    "::mlir::BoolAttr":$fp32_dest_acc_en,
    "::mlir::BoolAttr":$preserve_fp32_precision,
    "::mlir::BoolAttr":$math_approx_mode,
    ArrayRefParameter<"uint32_t">:$compile_args,
    "::mlir::DictionaryAttr":$defines);

  let assemblyFormat = "`<` $math_fidelity `,` $fp32_dest_acc_en `,` $preserve_fp32_precision `,` $math_approx_mode `,` `<` $compile_args `>` `,` $defines `>`";
}

def TT_ComputeAttributesAttr : TT_Attr<"ComputeAttributes", "compute_attributes"> {
  let summary = "Data movement attributes.";
  let description = [{
    An attribute representing the compute_attributes_t structure.
  }];

  let parameters = (ins 
    TT_CoreRangeAttr:$core_range,
    "::mlir::StringAttr":$kernel_path,
    TT_ComputeConfigAttr:$compute_config);
    // TODO runtime_args missing

  let assemblyFormat = "`<` $core_range `,` $kernel_path `,` $compute_config `>`";
}

// An array of compute attributes.
def TT_ComputeAttributesArrayAttr : TypedArrayAttrBase<TT_ComputeAttributesAttr, "">;


//===----------------------------------------------------------------------===//
// TT type definitions
//===----------------------------------------------------------------------===//

class TT_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TT_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def TT_Tile : TT_Type<"Tile", "tile", [MemRefElementTypeInterface]> {
    let summary = "TT tile";
    let description = "Tile type in TT dialect";
    let parameters = (ins ArrayRefParameter<"int64_t">:$shape, "DataType":$dataType);
    let assemblyFormat = "`<` custom<DimensionList>($shape) `,` $dataType `>`";

    let extraClassDeclaration = [{
      static TileType get(::mlir::MLIRContext *context, Type elementType, ArrayRef<int64_t> shape = {32, 32});
      SmallVector<int64_t> getScalarShape(SmallVector<int64_t> tiledShape) const;
      SmallVector<int64_t> getTiledShape(SmallVector<int64_t> scalarShape) const;
      uint64_t getSizeBytes() const;
      int64_t getHeight() const { return getShape()[0]; }
      int64_t getWidth() const { return getShape()[1]; }
      // Returns the scalar element type of the tile, if compressed it returns
      // the corresponding uncompressed element type, i.e. bfp_bf8 -> bf16
      Type getElementType() const;
    }];

    let genVerifyDecl = 1;
}

def TT_Device : TT_Type<"Device", "device", []> {
    let summary = "TT device";
    let description = "Device type in TT dialect";
    let parameters = (ins TT_DeviceAttr:$desc);
    let assemblyFormat = "`<` $desc `>`";
}

#endif
