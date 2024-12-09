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

def TTNN_TensorMemoryLayout_Interleaved : I32EnumAttrCase<"Interleaved", 1, "interleaved">;
def TTNN_TensorMemoryLayout_SingleBank : I32EnumAttrCase<"SingleBank", 2, "single_bank">;
def TTNN_TensorMemoryLayout_HeightSharded : I32EnumAttrCase<"HeightSharded", 3, "height_sharded">;
def TTNN_TensorMemoryLayout_WidthSharded : I32EnumAttrCase<"WidthSharded", 4, "width_sharded">;
def TTNN_TensorMemoryLayout_BlockSharded : I32EnumAttrCase<"BlockSharded", 5, "block_sharded">;

def TTNN_TensorMemoryLayout : I32EnumAttr<"TensorMemoryLayout", "TTNN Tensor Memory Layout",
                           [
                            TTNN_TensorMemoryLayout_Interleaved,
                            TTNN_TensorMemoryLayout_SingleBank,
                            TTNN_TensorMemoryLayout_HeightSharded,
                            TTNN_TensorMemoryLayout_WidthSharded,
                            TTNN_TensorMemoryLayout_BlockSharded,
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

def TTNN_ReduceType_Sum  : I32EnumAttrCase<"Sum",  0, "sum">;
def TTNN_ReduceType_Mean : I32EnumAttrCase<"Mean", 1, "mean">;
def TTNN_ReduceType_Max  : I32EnumAttrCase<"Max",  2, "max">;
def TTNN_ReduceType_Min  : I32EnumAttrCase<"Min",  3, "min">;
def TTNN_ReduceType_Std  : I32EnumAttrCase<"Std",  4, "std">;
def TTNN_ReduceType_Var  : I32EnumAttrCase<"Var",  5, "var">;

def TTNN_ReduceType: I32EnumAttr<"ReduceType", "TTNN Reduce Operation Type",
                          [
                           TTNN_ReduceType_Sum,
                           TTNN_ReduceType_Mean,
                           TTNN_ReduceType_Max,
                           TTNN_ReduceType_Min,
                           TTNN_ReduceType_Std,
                           TTNN_ReduceType_Var,
                          ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttnn::operations::reduction";
}

#endif
