// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTCORE_IR_TTCOREOPSENUMS_TD
#define TTMLIR_DIALECT_TTCORE_IR_TTCOREOPSENUMS_TD

include "mlir/IR/EnumAttr.td"

def TTCore_Float32 : I32EnumAttrCase<"Float32", 0, "f32">;
def TTCore_Float16 : I32EnumAttrCase<"Float16", 1, "f16">;
def TTCore_BFloat16 : I32EnumAttrCase<"BFloat16", 2, "bf16">;
def TTCore_BFP_Float8 : I32EnumAttrCase<"BFP_Float8", 3, "bfp_f8">;
def TTCore_BFP_BFloat8 : I32EnumAttrCase<"BFP_BFloat8", 4, "bfp_bf8">;
def TTCore_BFP_Float4 : I32EnumAttrCase<"BFP_Float4", 5, "bfp_f4">;
def TTCore_BFP_BFloat4 : I32EnumAttrCase<"BFP_BFloat4", 6, "bfp_bf4">;
def TTCore_BFP_Float2 : I32EnumAttrCase<"BFP_Float2", 7, "bfp_f2">;
def TTCore_BFP_BFloat2 : I32EnumAttrCase<"BFP_BFloat2", 8, "bfp_bf2">;
def TTCore_UInt32 : I32EnumAttrCase<"UInt32", 9, "u32">;
def TTCore_UInt16 : I32EnumAttrCase<"UInt16", 10, "u16">;
def TTCore_UInt8 : I32EnumAttrCase<"UInt8", 11, "u8">;
def TTCore_Int32: I32EnumAttrCase<"Int32", 12, "si32">;
def TTCore_Bool : I32EnumAttrCase<"Bool", 13, "i1">;

def TTCore_DataType : I32EnumAttr<"DataType", "TT DataTypes",
                           [
                            TTCore_Float32,
                            TTCore_Float16,
                            TTCore_BFloat16,
                            TTCore_BFP_Float8,
                            TTCore_BFP_BFloat8,
                            TTCore_BFP_Float4,
                            TTCore_BFP_BFloat4,
                            TTCore_BFP_Float2,
                            TTCore_BFP_BFloat2,
                            TTCore_UInt32,
                            TTCore_UInt16,
                            TTCore_UInt8,
                            TTCore_Int32,
                            TTCore_Bool
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttcore";
  let stringToSymbolFnName = "DataTypeStringToEnum";
  let symbolToStringFnName = "DataTypeEnumToString";
}

def TTCore_Grayskull : I32EnumAttrCase<"Grayskull", 0, "grayskull">;
def TTCore_WormholeB0 : I32EnumAttrCase<"WormholeB0", 1, "wormhole_b0">;
def TTCore_Blackhole : I32EnumAttrCase<"Blackhole", 2, "blackhole">;

def TTCore_Arch : I32EnumAttr<"Arch", "TT Arch",
                           [
                            TTCore_Grayskull,
                            TTCore_WormholeB0,
                            TTCore_Blackhole,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttcore";
}


def TTCore_System : I32EnumAttrCase<"System", 0, "system">;
def TTCore_SystemMMIO : I32EnumAttrCase<"SystemMMIO", 1, "mmio">;
def TTCore_DeviceDRAM : I32EnumAttrCase<"DeviceDRAM", 2, "dram">;
def TTCore_DeviceL1 : I32EnumAttrCase<"DeviceL1", 3, "l1">;
def TTCore_RegisterDst : I32EnumAttrCase<"RegisterDst", 4, "dst">;
def TTCore_DeviceRiscvL1 : I32EnumAttrCase<"DeviceRiscvL1", 5, "riscv_l1">;

def TTCore_MemorySpace : I32EnumAttr<"MemorySpace", "TT MemorySpace",
                           [
                            TTCore_System,
                            TTCore_SystemMMIO,
                            TTCore_DeviceDRAM,
                            TTCore_DeviceL1,
                            TTCore_RegisterDst,
                            TTCore_DeviceRiscvL1,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttcore";
}

def TTCore_Parallel : I32EnumAttrCase<"Parallel", 0, "parallel">;
def TTCore_Reduction : I32EnumAttrCase<"Reduction", 1, "reduction">;

def TTCore_IteratorType : I32EnumAttr<"IteratorType", "TT IteratorType",
                           [
                            TTCore_Parallel,
                            TTCore_Reduction,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttcore";
}

def TTCore_Undef : I32EnumAttrCase<"Undef", 0, "undef">;
def TTCore_Zero : I32EnumAttrCase<"Zero", 1, "zero">;
def TTCore_One : I32EnumAttrCase<"One", 2, "one">;
def TTCore_Inf : I32EnumAttrCase<"Inf", 3, "inf">;
def TTCore_NegInf : I32EnumAttrCase<"NegInf", 4, "neginf">;

def TTCore_OOBVal : I32EnumAttr<"OOBVal", "TT OOBVal",
                           [
                            TTCore_Undef,
                            TTCore_Zero,
                            TTCore_One,
                            TTCore_Inf,
                            TTCore_NegInf,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttcore";
}

def TTCore_ChipCapabilityHostMMIO : I32BitEnumAttrCaseBit<"HostMMIO", 0, "host_mmio">;

def TTCore_ChipCapability : I32BitEnumAttr<"ChipCapability", "TT Chip Capabilities",
                           [
                            TTCore_ChipCapabilityHostMMIO,
                           ]> {
  let genSpecializedAttr = 1;
  let cppNamespace = "::mlir::tt::ttcore";
  let separator = "|";
  let printBitEnumPrimaryGroups = 1;
}

def TTCore_ReduceType_Sum  : I32EnumAttrCase<"Sum",  0, "sum">;
def TTCore_ReduceType_Mean : I32EnumAttrCase<"Mean", 1, "mean">;
def TTCore_ReduceType_Max  : I32EnumAttrCase<"Max",  2, "max">;
def TTCore_ReduceType_Min  : I32EnumAttrCase<"Min",  3, "min">;
def TTCore_ReduceType_Std  : I32EnumAttrCase<"Std",  4, "std">;
def TTCore_ReduceType_Var  : I32EnumAttrCase<"Var",  5, "var">;

def TTCore_ReduceType: I32EnumAttr<"ReduceType", "TT Reduce Type",
                          [
                           TTCore_ReduceType_Sum,
                           TTCore_ReduceType_Mean,
                           TTCore_ReduceType_Max,
                           TTCore_ReduceType_Min,
                           TTCore_ReduceType_Std,
                           TTCore_ReduceType_Var,
                          ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttcore";
}

def TTCore_MeshShardDirection_FullToShard : I32EnumAttrCase<"FullToShard",  0, "full_to_shard">;
def TTCore_MeshShardDirection_ShardToFull : I32EnumAttrCase<"ShardToFull",  1, "shard_to_full">;

def TTCore_MeshShardDirection: I32EnumAttr<"MeshShardDirection", "TT MeshShardDirection",
                          [
                           TTCore_MeshShardDirection_FullToShard,
                           TTCore_MeshShardDirection_ShardToFull,
                          ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttcore";
}

def TTCore_MeshShardType_Identity  : I32EnumAttrCase<"Identity",  0, "identity">;
def TTCore_MeshShardType_Replicate : I32EnumAttrCase<"Replicate", 1, "replicate">;
def TTCore_MeshShardType_Maximal   : I32EnumAttrCase<"Maximal",   2, "maximal">;
def TTCore_MeshShardType_Devices   : I32EnumAttrCase<"Devices",   3, "devices">;

def TTCore_MeshShardType: I32EnumAttr<"MeshShardType", "TT MeshShardType",
                          [
                           TTCore_MeshShardType_Identity,
                           TTCore_MeshShardType_Replicate,
                           TTCore_MeshShardType_Maximal,
                           TTCore_MeshShardType_Devices,
                          ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttcore";
}

def TTCore_CPURoleHost : I32EnumAttrCase<"Host", 0, "host">;
def TTCore_CPURoleDevice : I32EnumAttrCase<"Device", 1, "device">;

def TTCore_CPURole : I32EnumAttr<"CPURole", "TT CPU Role",
                            [
                              TTCore_CPURoleHost,
                              TTCore_CPURoleDevice,
                            ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttcore";
}

def TTCore_ArgumentType_Input : I32EnumAttrCase<"Input", 0, "input">;
def TTCore_ArgumentType_Parameter: I32EnumAttrCase<"Parameter", 1, "parameter">;
def TTCore_ArgumentType_Constant : I32EnumAttrCase<"Constant", 2, "constant">;
def TTCore_ArgumentType_Default : I32EnumAttrCase<"Default", 3, "default">;

def TTCore_ArgumentType : I32EnumAttr<"ArgumentType", "Argument Type",
                            [
                              TTCore_ArgumentType_Input,
                              TTCore_ArgumentType_Parameter,
                              TTCore_ArgumentType_Constant,
                              TTCore_ArgumentType_Default,
                            ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttcore";
  let stringToSymbolFnName = "ArgumentTypeStringToEnum";
  let symbolToStringFnName = "ArgumentTypeEnumToString";
}

def TTCore_TensorMemoryLayout_Interleaved : I32EnumAttrCase<"Interleaved", 0, "interleaved">;
def TTCore_TensorMemoryLayout_Sharded     : I32EnumAttrCase<"Sharded",     1, "sharded">;

def TTCore_TensorMemoryLayout : I32EnumAttr<"TensorMemoryLayout", "TTCore Tensor Memory Layout",
                           [
                            TTCore_TensorMemoryLayout_Interleaved,
                            TTCore_TensorMemoryLayout_Sharded,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttcore";
}

def TTCore_ShardStatus_Presharded: I32EnumAttrCase<"Presharded", 0, "presharded">;
def TTCore_ShardStatus_Unsharded: I32EnumAttrCase<"Unsharded", 1, "unsharded">;

def TTCore_ShardStatus : I32EnumAttr<"ShardStatus", "Shard Status Type",
                            [
                              TTCore_ShardStatus_Presharded,
                              TTCore_ShardStatus_Unsharded,
                            ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttcore";
}

#endif // TTMLIR_DIALECT_TTCORE_IR_TTCOREOPSENUMS_TD
