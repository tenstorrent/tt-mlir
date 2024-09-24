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

def TTNN_BufferType_DRAM : I32EnumAttrCase<"DRAM", 0, "dram">;
def TTNN_BufferType_L1 : I32EnumAttrCase<"L1", 1, "l1">;
def TTNN_BufferType_SystemMemory : I32EnumAttrCase<"SystemMemory", 2, "system_memory">;
def TTNN_BufferType_L1Small : I32EnumAttrCase<"L1Small", 3, "l1_small">;
def TTNN_BufferType_Trace : I32EnumAttrCase<"Trace", 4, "trace">;

def TTNN_BufferType : I32EnumAttr<"BufferType", "TTNN Buffer Type",
                           [
                            TTNN_BufferType_DRAM,
                            TTNN_BufferType_L1,
                            TTNN_BufferType_SystemMemory,
                            TTNN_BufferType_L1Small,
                            TTNN_BufferType_Trace,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttnn";
}

#endif
