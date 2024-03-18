// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.     ■ Too many errors emitted, stopping now
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_TTOPSENUMS_TD
#define TTMLIR_TTMLIR_TTOPSENUMS_TD

include "mlir/IR/EnumAttr.td"

def TT_Float32 : I32EnumAttrCase<"Float32", 0>;
def TT_BFloat16 : I32EnumAttrCase<"BFloat16", 1>;

def TT_DataType : I32EnumAttr<"DataType", "TT DataTypes",
                           [TT_Float32, TT_BFloat16]> {
  let cppNamespace = "::mlir::tt";
  let stringToSymbolFnName = "ConvertToEnum";
  let symbolToStringFnName = "ConvertToString";
}

#endif
