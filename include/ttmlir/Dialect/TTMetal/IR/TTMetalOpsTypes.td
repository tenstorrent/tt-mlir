// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALOPSTYPES_TD
#define TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALOPSTYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "ttmlir/Dialect/TTMetal/IR/TTMetalBase.td"
include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsEnums.td"

//===----------------------------------------------------------------------===//
// TTMetal type definitions
//===----------------------------------------------------------------------===//

class TTMetal_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TTMetal_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// TTMetal attr definitions
//===----------------------------------------------------------------------===//
// Should Attr be a separate file?

class TTMetal_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Attribute">
    : AttrDef<TTMetal_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
  let attrName = "ttmetal." # attrMnemonic;
}

def TTMetal_CoreRangeAttr : TTMetal_Attr<"CoreRange", "core_range"> {
  let summary = "TTMetal grid attribute";
  let description = [{
    TTMetal grid attribute
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

def TTMetal_CoreRangeArrayAttr : TypedArrayAttrBase<TTMetal_CoreRangeAttr, "">;

#endif
