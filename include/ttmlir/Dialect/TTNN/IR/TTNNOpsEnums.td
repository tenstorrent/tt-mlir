// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSENUMS_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSENUMS_TD

include "mlir/IR/EnumAttr.td"

def TTNN_Layout_RowMajor : I32EnumAttrCase<"RowMajor", 0, "row_major">;
def TTNN_Layout_Tile : I32EnumAttrCase<"Tile", 1, "tile">;
def TTNN_Layout_Invalid : I32EnumAttrCase<"Invalid", 2, "invalid">;

def TTNN_Layout : I32EnumAttr<"Layout", "TTNN Layout",
                           [
                            TTNN_Layout_RowMajor,
                            TTNN_Layout_Tile,
                            TTNN_Layout_Invalid,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttnn";
}

#endif
