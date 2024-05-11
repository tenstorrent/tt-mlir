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

def TT_System : I32EnumAttrCase<"System", 0, "system">;
def TT_DRAM : I32EnumAttrCase<"DRAM", 1, "dram">;
def TT_L1 : I32EnumAttrCase<"L1", 2, "l1">;

def TT_MemorySpace : I32EnumAttr<"MemorySpace", "TT MemorySpace",
                           [
                            TT_System,
                            TT_DRAM,
                            TT_L1,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt";
}

def TT_Parallel : I32EnumAttrCase<"Parallel", 0, "parallel">;
def TT_Systolic : I32EnumAttrCase<"Systolic", 1, "systolic">;
def TT_Broadcast : I32EnumAttrCase<"Broadcast", 2, "broadcast">;

def TT_IteratorType : I32EnumAttr<"IteratorType", "TT IteratorType",
                           [
                            TT_Parallel,
                            TT_Systolic,
                            TT_Broadcast,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt";
}

def TT_Undef : I32EnumAttrCase<"Undef", 0, "undef">;
def TT_Zero : I32EnumAttrCase<"Zero", 1, "zero">;
def TT_One : I32EnumAttrCase<"One", 2, "one">;
def TT_Inf : I32EnumAttrCase<"Inf", 3, "inf">;
def TT_NegInf : I32EnumAttrCase<"NegInf", 4, "neginf">;

def TT_OOBVal : I32EnumAttr<"OOBVal", "TT OOBVal",
                           [
                            TT_Undef,
                            TT_Zero,
                            TT_One,
                            TT_Inf,
                            TT_NegInf,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt";
}

def TT_OperandConstraintDRAM : I32BitEnumAttrCaseBit<"DRAM", 0, "dram">;
def TT_OperandConstraintL1 : I32BitEnumAttrCaseBit<"L1", 1, "l1">;
def TT_OperandConstraintScalar : I32BitEnumAttrCaseBit<"Scalar", 2, "scalar">;
def TT_OperandConstraintTile : I32BitEnumAttrCaseBit<"Tile", 3, "tile">;
def TT_OperandConstraintAny : I32BitEnumAttrCaseGroup<"Any", [TT_OperandConstraintDRAM, TT_OperandConstraintL1, TT_OperandConstraintScalar, TT_OperandConstraintTile], "any">;
def TT_OperandConstraintAnyTile : I32BitEnumAttrCaseGroup<"AnyTile", [TT_OperandConstraintDRAM, TT_OperandConstraintL1, TT_OperandConstraintTile], "any_tile">;

def TT_OperandConstraint : I32BitEnumAttr<"OperandConstraint", "TT Operand Constraints",
                           [
                            TT_OperandConstraintDRAM,
                            TT_OperandConstraintL1,
                            TT_OperandConstraintScalar,
                            TT_OperandConstraintTile,
                            TT_OperandConstraintAny,
                            TT_OperandConstraintAnyTile,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt";
}

#endif
