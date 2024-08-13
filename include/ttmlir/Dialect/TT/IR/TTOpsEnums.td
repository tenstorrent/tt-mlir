// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_TTOPSENUMS_TD
#define TTMLIR_TTMLIR_TTOPSENUMS_TD

include "mlir/IR/EnumAttr.td"

def TT_Float32 : I32EnumAttrCase<"Float32", 0, "f32">;
def TT_Float16 : I32EnumAttrCase<"Float16", 1, "f16">;
def TT_BFloat16 : I32EnumAttrCase<"BFloat16", 2, "bf16">;
def TT_BFP_Float8 : I32EnumAttrCase<"BFP_Float8", 3, "bfp_f8">;
def TT_BFP_BFloat8 : I32EnumAttrCase<"BFP_BFloat8", 4, "bfp_bf8">;
def TT_BFP_Float4 : I32EnumAttrCase<"BFP_Float4", 5, "bfp_f4">;
def TT_BFP_BFloat4 : I32EnumAttrCase<"BFP_BFloat4", 6, "bfp_bf4">;
def TT_BFP_Float2 : I32EnumAttrCase<"BFP_Float2", 7, "bfp_f2">;
def TT_BFP_BFloat2 : I32EnumAttrCase<"BFP_BFloat2", 8, "bfp_bf2">;
def TT_UInt32 : I32EnumAttrCase<"UInt32", 9, "u32">;
def TT_UInt16 : I32EnumAttrCase<"UInt16", 10, "u16">;
def TT_UInt8 : I32EnumAttrCase<"UInt8", 11, "u8">;

def TT_DataType : I32EnumAttr<"DataType", "TT DataTypes",
                           [
                            TT_Float32,
                            TT_Float16,
                            TT_BFloat16,
                            TT_BFP_Float8,
                            TT_BFP_BFloat8,
                            TT_BFP_Float4,
                            TT_BFP_BFloat4,
                            TT_BFP_Float2,
                            TT_BFP_BFloat2,
                            TT_UInt32,
                            TT_UInt16,
                            TT_UInt8
                           ]> {
  let cppNamespace = "::mlir::tt";
  let stringToSymbolFnName = "DataTypeStringToEnum";
  let symbolToStringFnName = "DataTypeEnumToString";
}

def TT_Grayskull : I32EnumAttrCase<"Grayskull", 0, "grayskull">;
def TT_WormholeB0 : I32EnumAttrCase<"WormholeB0", 1, "wormhole_b0">;
def TT_Blackhole : I32EnumAttrCase<"Blackhole", 2, "blackhole">;

def TT_Arch : I32EnumAttr<"Arch", "TT Arch",
                           [
                            TT_Grayskull,
                            TT_WormholeB0,
                            TT_Blackhole,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt";
}


def TT_System : I32EnumAttrCase<"System", 0, "system">;
def TT_SystemMMIO : I32EnumAttrCase<"SystemMMIO", 1, "mmio">;
def TT_DeviceDRAM : I32EnumAttrCase<"DeviceDRAM", 2, "dram">;
def TT_DeviceL1 : I32EnumAttrCase<"DeviceL1", 3, "l1">;

def TT_MemorySpace : I32EnumAttr<"MemorySpace", "TT MemorySpace",
                           [
                            TT_System,
                            TT_SystemMMIO,
                            TT_DeviceDRAM,
                            TT_DeviceL1,
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

def TT_OperandConstraintSystem : I32BitEnumAttrCaseBit<"System", 0, "system">;
def TT_OperandConstraintDRAM : I32BitEnumAttrCaseBit<"DRAM", 1, "dram">;
def TT_OperandConstraintL1 : I32BitEnumAttrCaseBit<"L1", 2, "l1">;
def TT_OperandConstraintScalar : I32BitEnumAttrCaseBit<"Scalar", 3, "scalar">;
def TT_OperandConstraintTile : I32BitEnumAttrCaseBit<"Tile", 4, "tile">;
def TT_OperandConstraintAny : I32BitEnumAttrCaseGroup<"Any", [TT_OperandConstraintSystem, TT_OperandConstraintDRAM, TT_OperandConstraintL1, TT_OperandConstraintScalar, TT_OperandConstraintTile], "any">;
def TT_OperandConstraintAnyDevice : I32BitEnumAttrCaseGroup<"AnyDevice", [TT_OperandConstraintDRAM, TT_OperandConstraintL1, TT_OperandConstraintScalar, TT_OperandConstraintTile], "any_device">;
def TT_OperandConstraintAnyDeviceTile : I32BitEnumAttrCaseGroup<"AnyDeviceTile", [TT_OperandConstraintDRAM, TT_OperandConstraintL1, TT_OperandConstraintTile], "any_device_tile">;

def TT_OperandConstraint : I32BitEnumAttr<"OperandConstraint", "TT Operand Constraints",
                           [
                            TT_OperandConstraintSystem,
                            TT_OperandConstraintDRAM,
                            TT_OperandConstraintL1,
                            TT_OperandConstraintScalar,
                            TT_OperandConstraintTile,
                            TT_OperandConstraintAny,
                            TT_OperandConstraintAnyDevice,
                            TT_OperandConstraintAnyDeviceTile,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt";
}

def TT_ChipCapabilityNone : I32BitEnumAttrCaseBit<"None", 0, "none">;
def TT_ChipCapabilityPCIE : I32BitEnumAttrCaseBit<"PCIE", 1, "pcie">;
def TT_ChipCapabilityHostMMIO : I32BitEnumAttrCaseBit<"HostMMIO", 2, "host_mmio">;

def TT_ChipCapability : I32BitEnumAttr<"ChipCapability", "TT Chip Capabilities",
                           [
                            TT_ChipCapabilityNone,
                            TT_ChipCapabilityPCIE,
                            TT_ChipCapabilityHostMMIO,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt";
}

#endif
