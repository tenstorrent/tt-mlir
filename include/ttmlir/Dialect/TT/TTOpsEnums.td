// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_TTOPSENUMS_TD
#define TTMLIR_TTMLIR_TTOPSENUMS_TD

include "mlir/IR/EnumAttr.td"

def TT_Float32 : I32EnumAttrCase<"Float32", 0, "f32">;
def TT_Float16 : I32EnumAttrCase<"Float16", 1, "f16">;
def TT_BFloat16 : I32EnumAttrCase<"BFloat16", 2, "bf16">;
def TT_BC_Float8 : I32EnumAttrCase<"BC_Float8", 3, "bc_f8">;
def TT_BC_BFloat8 : I32EnumAttrCase<"BC_BFloat8", 4, "bc_bf8">;
def TT_BC_Float4 : I32EnumAttrCase<"BC_Float4", 5, "bc_f4">;
def TT_BC_BFloat4 : I32EnumAttrCase<"BC_BFloat4", 6, "bc_bf4">;
def TT_BC_Float2 : I32EnumAttrCase<"BC_Float2", 7, "bc_f2">;
def TT_BC_BFloat2 : I32EnumAttrCase<"BC_BFloat2", 8, "bc_bf2">;
def TT_UInt32 : I32EnumAttrCase<"UInt32", 9, "u32">;
def TT_UInt16 : I32EnumAttrCase<"UInt16", 10, "u16">;
def TT_UInt8 : I32EnumAttrCase<"UInt8", 11, "u8">;

def TT_DataType : I32EnumAttr<"DataType", "TT DataTypes",
                           [
                            TT_Float32,
                            TT_Float16,
                            TT_BFloat16,
                            TT_BC_Float8,
                            TT_BC_BFloat8,
                            TT_BC_Float4,
                            TT_BC_BFloat4,
                            TT_BC_Float2,
                            TT_BC_BFloat2,
                            TT_UInt32,
                            TT_UInt16,
                            TT_UInt8
                           ]> {
  let cppNamespace = "::mlir::tt";
  let stringToSymbolFnName = "DataTypeStringToEnum";
  let symbolToStringFnName = "DataTypeEnumToString";
}

def TT_Parallel : I32EnumAttrCase<"Parallel", 0, "parallel">;
def TT_Systolic : I32EnumAttrCase<"Systolic", 1, "systolic">;
def TT_Broadcast : I32EnumAttrCase<"Broadcast", 2, "Broadcast">;

def TT_IteratorType : I32EnumAttr<"IteratorType", "TT IteratorType",
                           [
                            TT_Parallel,
                            TT_Systolic,
                            TT_Broadcast,
                           ]> {
  let cppNamespace = "::mlir::tt";
  let stringToSymbolFnName = "IteratorTypeStringToEnum";
  let symbolToStringFnName = "IteratorTypeEnumToString";
}

#endif
