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
      static GridAttr get(::mlir::MLIRContext *context)
      {
        return GridAttr::get(context, {1, 1});
      }

      static GridAttr get(::mlir::MLIRContext *context, ArrayRef<int64_t> shape)
      {
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

  let parameters = (ins "ArchAttr":$arch, TT_GridAttr:$grid, "unsigned":$l1Size, "unsigned":$numDramChannels, "unsigned":$dramChannelSize);
  let assemblyFormat = "`{` `arch` `=` $arch `,` `grid` `=` $grid `,` `l1_size` `=` $l1Size `,` `num_dram_channels` `=` $numDramChannels `,` `dram_channel_size` `=` $dramChannelSize `}`";
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
  let summary = "TT layout attribute";
  let description = [{
    TT layout attribute
  }];

  let parameters = (ins ArrayRefParameter<"int64_t">:$strides,
                        "OOBVal":$oob_val,
                        "GridAttr":$grid,
                        "MemRefType":$memref);
  let assemblyFormat = "`<` custom<DimensionList>($strides)`,` $oob_val`,` $grid`,` $memref `>`";

  let extraClassDeclaration = [{
      MemorySpace getMemorySpace() const;
      bool isSystemMemorySpace() const { return ::mlir::tt::isSystemMemorySpace(getMemorySpace()); }
      bool isDeviceMemorySpace() const { return ::mlir::tt::isDeviceMemorySpace(getMemorySpace()); }
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
}

def TT_Device : TT_Type<"Device", "device", []> {
    let summary = "TT device";
    let description = "Device type in TT dialect";
    let parameters = (ins TT_GridAttr:$mesh, ArrayRefParameter<"unsigned">:$chipIds);
    let assemblyFormat = "`<` qualified($mesh) `,` `[` $chipIds `]` `>`";
}

#endif
