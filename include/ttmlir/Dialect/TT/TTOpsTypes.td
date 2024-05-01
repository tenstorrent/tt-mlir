// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_TTOPSTYPES_TD
#define TTMLIR_TTMLIR_TTOPSTYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
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

def TT_Grid : TT_Type<"Grid", "grid"> {
    let summary = "TT grid";
    let description = "Grid type in TT dialect";
    let parameters = (ins ArrayRefParameter<"int64_t">:$shape, "AffineMap":$physical_grid_mapping);
    let assemblyFormat = "`<` custom<GridDimension>($shape)`,` $physical_grid_mapping `>`";

    let extraClassDeclaration = [{
        static GridType unit(::mlir::MLIRContext *context) {
            return GridType::get(context, {1, 1}, AffineMap::getMultiDimIdentityMap(2, context));
        }
    }];
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

  let parameters = (ins "GridType":$value);
  let assemblyFormat = "`<` $value `>`";
}

def TT_IteratorTypeArrayAttr : TypedArrayAttrBase<TT_IteratorType, "">;

#endif
