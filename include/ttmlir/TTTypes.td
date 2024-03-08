// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.     ■ Too many errors emitted, stopping now
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_TTTYPES_TD
#define TTMLIR_TTMLIR_TTTYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "ttmlir/TTDialect.td"

//===----------------------------------------------------------------------===//
// TT type definitions
//===----------------------------------------------------------------------===//

class TT_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TT_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def TT_CustomType : TT_Type<"Custom", "custom"> {
    let summary = "TT custom type";
    let description = "Custom type in TT dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

#endif
