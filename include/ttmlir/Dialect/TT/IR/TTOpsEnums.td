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

def TT_ChipCapabilityPCIE : I32BitEnumAttrCaseBit<"PCIE", 0, "pcie">;
def TT_ChipCapabilityHostMMIO : I32BitEnumAttrCaseBit<"HostMMIO", 1, "host_mmio">;

def TT_ChipCapability : I32BitEnumAttr<"ChipCapability", "TT Chip Capabilities",
                           [
                            TT_ChipCapabilityPCIE,
                            TT_ChipCapabilityHostMMIO,
                           ]> {
  let genSpecializedAttr = 1;
  let cppNamespace = "::mlir::tt";
}

def TT_BufferAccessAlias : I32BitEnumAttrCaseBit<"Alias", 0, "alias">;
def TT_BufferAccessStream : I32BitEnumAttrCaseBit<"Stream", 1, "stream">;

def TT_BufferAccess : I32BitEnumAttr<"BufferAccess", "TT Buffer Access",
                           [
                            TT_BufferAccessAlias ,
                            TT_BufferAccessStream,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt";
}

def TT_Reader : I32EnumAttrCase<"Reader", 0, "reader">;
def TT_Writer : I32EnumAttrCase<"Writer", 1, "writer">;

def TT_DataMovementType : I32EnumAttr<"DataMovementType", "TTIR DataMovementType",
                           [
                            TT_Reader,
                            TT_Writer,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt";
}

def TT_LoFi : I32EnumAttrCase<"LoFi", 0, "lofi">;
def TT_HiFi2 : I32EnumAttrCase<"HiFi2", 1, "hifi2">;
def TT_HiFi3 : I32EnumAttrCase<"HiFi3", 2, "hifi3">;
def TT_HiFi4 : I32EnumAttrCase<"HiFi4", 3, "hifi4">;
def TT_Invalid : I32EnumAttrCase<"Invalid", 4, "invalid">;

def TT_MathFidelity : I32EnumAttr<"MathFidelity", "TTIR MathFidelity",
                           [
                            TT_LoFi,
                            TT_HiFi2,
                            TT_HiFi3,
                            TT_HiFi4,
                            TT_Invalid
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt";
}

def TT_CB_c_in0 : I32EnumAttrCase<"c_in0", 0, "c_in0">;
def TT_CB_c_in1 : I32EnumAttrCase<"c_in1", 1, "c_in1">;
def TT_CB_c_in2 : I32EnumAttrCase<"c_in2", 2, "c_in2">;
def TT_CB_c_in3 : I32EnumAttrCase<"c_in3", 3, "c_in3">;
def TT_CB_c_in4 : I32EnumAttrCase<"c_in4", 4, "c_in4">;
def TT_CB_c_in5 : I32EnumAttrCase<"c_in5", 5, "c_in5">;
def TT_CB_c_in6 : I32EnumAttrCase<"c_in6", 6, "c_in6">;
def TT_CB_c_in7 : I32EnumAttrCase<"c_in7", 7, "c_in7">;

def TT_CB_dataflow0 : I32EnumAttrCase<"dataflow0", 8, "dataflow0">;
def TT_CB_dataflow1 : I32EnumAttrCase<"dataflow1", 9, "dataflow1">;
def TT_CB_dataflow2 : I32EnumAttrCase<"dataflow2", 10, "dataflow2">;
def TT_CB_dataflow3 : I32EnumAttrCase<"dataflow3", 11, "dataflow3">;
def TT_CB_dataflow4 : I32EnumAttrCase<"dataflow4", 12, "dataflow4">;
def TT_CB_dataflow5 : I32EnumAttrCase<"dataflow5", 13, "dataflow5">;
def TT_CB_dataflow6 : I32EnumAttrCase<"dataflow6", 14, "dataflow6">;
def TT_CB_dataflow7 : I32EnumAttrCase<"dataflow7", 15, "dataflow7">;

def TT_CB_c_out0 : I32EnumAttrCase<"c_out0", 16, "c_out0">;
def TT_CB_c_out1 : I32EnumAttrCase<"c_out1", 17, "c_out1">;
def TT_CB_c_out2 : I32EnumAttrCase<"c_out2", 18, "c_out2">;
def TT_CB_c_out3 : I32EnumAttrCase<"c_out3", 19, "c_out3">;
def TT_CB_c_out4 : I32EnumAttrCase<"c_out4", 20, "c_out4">;
def TT_CB_c_out5 : I32EnumAttrCase<"c_out5", 21, "c_out5">;
def TT_CB_c_out6 : I32EnumAttrCase<"c_out6", 22, "c_out6">;
def TT_CB_c_out7 : I32EnumAttrCase<"c_out7", 23, "c_out7">;

def TT_CB_c_intermed0 : I32EnumAttrCase<"c_intermed0", 24, "c_intermed0">;
def TT_CB_c_intermed1 : I32EnumAttrCase<"c_intermed1", 25, "c_intermed1">;
def TT_CB_c_intermed2 : I32EnumAttrCase<"c_intermed2", 26, "c_intermed2">;
def TT_CB_c_intermed3 : I32EnumAttrCase<"c_intermed3", 27, "c_intermed3">;
def TT_CB_c_intermed4 : I32EnumAttrCase<"c_intermed4", 28, "c_intermed4">;
def TT_CB_c_intermed5 : I32EnumAttrCase<"c_intermed5", 29, "c_intermed5">;
def TT_CB_c_intermed6 : I32EnumAttrCase<"c_intermed6", 30, "c_intermed6">;
def TT_CB_c_intermed7 : I32EnumAttrCase<"c_intermed7", 31, "c_intermed7">;

// NOTE: TTKernel_CB already exists, think about how to consolidate definitions.
def TT_CB : I32EnumAttr<"CB", "CB Enumeration",
                           [
                            TT_CB_c_in0,
                            TT_CB_c_in1,
                            TT_CB_c_in2,
                            TT_CB_c_in3,
                            TT_CB_c_in4,
                            TT_CB_c_in5,
                            TT_CB_c_in6,
                            TT_CB_c_in7,
                            TT_CB_dataflow0,
                            TT_CB_dataflow1,
                            TT_CB_dataflow2,
                            TT_CB_dataflow3,
                            TT_CB_dataflow4,
                            TT_CB_dataflow5,
                            TT_CB_dataflow6,
                            TT_CB_dataflow7,
                            TT_CB_c_out0,
                            TT_CB_c_out1,
                            TT_CB_c_out2,
                            TT_CB_c_out3,
                            TT_CB_c_out4,
                            TT_CB_c_out5,
                            TT_CB_c_out6,
                            TT_CB_c_out7,
                            TT_CB_c_intermed0,
                            TT_CB_c_intermed1,
                            TT_CB_c_intermed2,
                            TT_CB_c_intermed3,
                            TT_CB_c_intermed4,
                            TT_CB_c_intermed5,
                            TT_CB_c_intermed6,
                            TT_CB_c_intermed7
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt";
}

def TT_TensorAddr : I32EnumAttrCase<"TensorAddr", 0, "tensor_addr">;

def TT_RuntimeArgumentType : I32EnumAttr<"RuntimeArgumentType", "TTIR Runtime Argument type",
                           [
                            TT_TensorAddr
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt";
}

#endif
