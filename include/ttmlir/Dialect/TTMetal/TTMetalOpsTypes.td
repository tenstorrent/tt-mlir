// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALOPSTYPES_TD
#define TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALOPSTYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "ttmlir/Dialect/TTMetal/TTMetalDialect.td"
include "ttmlir/Dialect/TTMetal/TTMetalOpsEnums.td"

//===----------------------------------------------------------------------===//
// TTMetal type definitions
//===----------------------------------------------------------------------===//

class TTMetal_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TTMetal_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def TTMetal_Tmp : TTMetal_Type<"Tmp", "tmp"> {
    let summary = "TTMetal tmp";
    let description = "Tmp type in TTMetal dialect";
    let parameters = (ins "unsigned":$tmp);
    let assemblyFormat = "`<` $tmp `>`";
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

def TTMetal_ThreadAttr : EnumAttr<TTMetal_Dialect, TTMetal_Thread, "operand_constraint"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTMetal_ThreadArrayAttr : TypedArrayAttrBase<TTMetal_ThreadAttr, "">;

#endif
