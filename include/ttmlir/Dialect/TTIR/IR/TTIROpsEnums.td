// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTIR_ENUMS_TD
#define TTMLIR_TTIR_ENUMS_TD

include "mlir/IR/EnumAttr.td"

def TTIR_PoolingMethodAverage : I32EnumAttrCase<"Average", 0>;
def TTIR_PoolingMethodMax : I32EnumAttrCase<"Max", 1>;

def TTIR_PoolingMethod : I32EnumAttr<"PoolingMethod", "TTIR PoolingMethod", [
                                      TTIR_PoolingMethodAverage,
                                      TTIR_PoolingMethodMax
                                    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttir";
}

#endif
