// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALOPSENUMS_TD
#define TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALOPSENUMS_TD

include "mlir/IR/EnumAttr.td"

def TTMetal_Noc0 : I32EnumAttrCase<"Noc0", 0, "noc0">;
def TTMetal_Noc1 : I32EnumAttrCase<"Noc1", 1, "noc1">;
def TTMetal_Tensix : I32EnumAttrCase<"Tensix", 2, "tensix">;
def TTMetal_Ethernet : I32EnumAttrCase<"Ethernet", 3, "ethernet">;

def TTMetal_Thread : I32EnumAttr<"Thread", "TTMetal DataTypes",
                           [
                            TTMetal_Noc0,
                            TTMetal_Noc1,
                            TTMetal_Tensix,
                            TTMetal_Ethernet,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttmetal";
}

#endif
