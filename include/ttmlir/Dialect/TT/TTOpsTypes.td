// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_TTOPSTYPES_TD
#define TTMLIR_TTMLIR_TTOPSTYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "ttmlir/Dialect/TT/TTDialect.td"
include "ttmlir/Dialect/TT/TTOpsEnums.td"

//===----------------------------------------------------------------------===//
// TT type definitions
//===----------------------------------------------------------------------===//

class TT_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TT_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def TT_Tile : TT_Type<"Tile", "tile"> {
    let summary = "TT tile";
    let description = "Tile type in TT dialect";
    let parameters = (ins "unsigned":$height, "unsigned":$width, "DataType":$dataType);
    let assemblyFormat = "`<` $height `x` $width`,` $dataType `>`";
}

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
        return GridAttr::get(context, {1, 1}, AffineMap::get(context));
      }
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

#endif
