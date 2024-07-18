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
                          "$_builder.getEmptyAffineMap()">:$physical_grid_mapping);
  let assemblyFormat = "`<` custom<DimensionList>($shape) (`,` $physical_grid_mapping^)? `>`";

  let extraClassDeclaration = [{
      static GridAttr get(::mlir::MLIRContext *context) {
        return GridAttr::get(context, {1, 1});
      }

      static GridAttr get(::mlir::MLIRContext *context, ArrayRef<int64_t> shape) {
        return GridAttr::get(context, shape, AffineMap::get(context));
      }
  }];
}

def TT_ChipCapabilityAttr : EnumAttr<TT_Dialect, TT_ChipCapability, "Chip_capability"> {
  let assemblyFormat = "`<` $value `>`";
}

def TT_ArchAttr : EnumAttr<TT_Dialect, TT_Arch, "arch"> {
  let assemblyFormat = "`<` $value `>`";
}

def TT_ChipDescAttr : TT_Attr<"ChipDesc", "chip_desc"> {
  let summary = "TT chip_desc attribute";
  let description = [{
    TT chip_desc attribute
  }];

  let parameters = (ins "ArchAttr":$arch, TT_GridAttr:$grid, "unsigned":$l1Size, "unsigned":$numDramChannels, "unsigned":$dramChannelSize, "unsigned":$nocL1AddressAlignBytes, "unsigned":$pcieAddressAlignBytes, "unsigned":$nocDRAMAddressAlignBytes);
  let assemblyFormat = "`{` `arch` `=` $arch `,` `grid` `=` $grid `,` `l1_size` `=` $l1Size `,` `num_dram_channels` `=` $numDramChannels `,` `dram_channel_size` `=` $dramChannelSize `,` `noc_l1_address_align_bytes` `=` $nocL1AddressAlignBytes `,` `pcie_address_align_bytes` `=` $pcieAddressAlignBytes `,` `noc_dram_address_align_bytes` `=` $nocDRAMAddressAlignBytes `}`";
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

  let parameters = (ins "unsigned":$endpoint0, "unsigned":$endpoint1);
  let assemblyFormat = "`<` $endpoint0 `,` $endpoint1 `>`";
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
      MemorySpace getMemorySpace() const;
      bool isSystemMemorySpace() const { return ::mlir::tt::isSystemMemorySpace(getMemorySpace()); }
      bool isDeviceMemorySpace() const { return ::mlir::tt::isDeviceMemorySpace(getMemorySpace()); }
      Type getElementType() const;
      llvm::SmallVector<int64_t> getStride() const;
      llvm::SmallVector<int64_t> getShardShape() const;
      LayoutAttr withGrid(::mlir::MLIRContext *context, ArrayRef<int64_t> tensorShape, GridAttr grid, ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals = {{0, -1}});
      LayoutAttr withGrid(::mlir::MLIRContext *context,
                          RankedTensorType ty,
                          GridAttr grid,
                          ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals = {{0, -1}});
      LayoutAttr withElementType(::mlir::MLIRContext *context, Type elementType);
  }];
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
    let parameters = (ins "unsigned":$height, "unsigned":$width, "DataType":$dataType);
    let assemblyFormat = "`<` $height `x` $width`,` $dataType `>`";

    let extraClassDeclaration = [{
      SmallVector<int64_t> getScalarShape(SmallVector<int64_t> tiledShape) const;
      SmallVector<int64_t> getTiledShape(SmallVector<int64_t> scalarShape) const;
    }];
}

def TT_Device : TT_Type<"Device", "device", []> {
    let summary = "TT device";
    let description = "Device type in TT dialect";
    let parameters = (ins TT_GridAttr:$mesh, ArrayRefParameter<"unsigned">:$chipIds);
    let assemblyFormat = "`<` qualified($mesh) `,` `[` $chipIds `]` `>`";
}

#endif
