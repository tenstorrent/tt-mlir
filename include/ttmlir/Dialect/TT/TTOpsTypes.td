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

  let parameters = (ins ArrayRefParameter<"int64_t">:$shape, "AffineMap":$physical_grid_mapping);
  let assemblyFormat = "`<` custom<DimensionList>($shape)`,` $physical_grid_mapping `>`";

  let extraClassDeclaration = [{
      static GridAttr unit(::mlir::MLIRContext *context) {
          return GridAttr::get(context, {1, 1}, AffineMap::getMultiDimIdentityMap(2, context));
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
}

def TT_IteratorTypeArrayAttr : TypedArrayAttrBase<TT_IteratorType, "">;

#endif
