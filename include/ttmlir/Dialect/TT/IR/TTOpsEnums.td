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
  let genSpecializedAttr = 0;
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
def TT_Reduction : I32EnumAttrCase<"Reduction", 3, "reduction">;

def TT_IteratorType : I32EnumAttr<"IteratorType", "TT IteratorType",
                           [
                            TT_Parallel,
                            TT_Systolic,
                            TT_Broadcast,
                            TT_Reduction,
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

def TT_StreamModeAlias : I32BitEnumAttrCaseBit<"Alias", 0, "alias">;
def TT_StreamModeStream : I32BitEnumAttrCaseBit<"Stream", 1, "stream">;

def TT_StreamMode : I32BitEnumAttr<"StreamMode", "TT Stream Mode",
                           [
                            TT_StreamModeAlias,
                            TT_StreamModeStream,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt";
}

def TT_ReduceType_Sum  : I32EnumAttrCase<"Sum",  0, "sum">;
def TT_ReduceType_Mean : I32EnumAttrCase<"Mean", 1, "mean">;
def TT_ReduceType_Max  : I32EnumAttrCase<"Max",  2, "max">;
def TT_ReduceType_Min  : I32EnumAttrCase<"Min",  3, "min">;
def TT_ReduceType_Std  : I32EnumAttrCase<"Std",  4, "std">;
def TT_ReduceType_Var  : I32EnumAttrCase<"Var",  5, "var">;

def TT_ReduceType: I32EnumAttr<"ReduceType", "TT Reduce Type",
                          [
                           TT_ReduceType_Sum,
                           TT_ReduceType_Mean,
                           TT_ReduceType_Max,
                           TT_ReduceType_Min,
                           TT_ReduceType_Std,
                           TT_ReduceType_Var,
                          ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt";
}

def TT_MeshShardDirection_FullToShard : I32EnumAttrCase<"FullToShard",  0, "full_to_shard">;
def TT_MeshShardDirection_ShardToFull : I32EnumAttrCase<"ShardToFull",  1, "shard_to_full">;

def TT_MeshShardDirection: I32EnumAttr<"MeshShardDirection", "TT MeshShardDirection",
                          [
                           TT_MeshShardDirection_FullToShard,
                           TT_MeshShardDirection_ShardToFull,
                          ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt";
}

def TT_MeshShardType_Manual    : I32EnumAttrCase<"Manual",    0, "manual">;
def TT_MeshShardType_Replicate : I32EnumAttrCase<"Replicate", 1, "replicate">;
def TT_MeshShardType_Maximal   : I32EnumAttrCase<"Maximal",   2, "maximal">;
def TT_MeshShardType_Devices   : I32EnumAttrCase<"Devices",   3, "devices">;

def TT_MeshShardType: I32EnumAttr<"MeshShardType", "TT MeshShardType",
                          [
                           TT_MeshShardType_Manual,
                           TT_MeshShardType_Replicate,
                           TT_MeshShardType_Maximal,
                           TT_MeshShardType_Devices,
                          ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt";
}

def TT_CPURoleHost : I32EnumAttrCase<"Host", 0, "host">;
def TT_CPURoleDevice : I32EnumAttrCase<"Device", 1, "device">;

def TT_CPURole : I32EnumAttr<"CPURole", "TT CPU Role",
                            [
                              TT_CPURoleHost,
                              TT_CPURoleDevice,
                            ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt";
}

def TT_ArgumentType_Input : I32EnumAttrCase<"Input", 0, "input">;
def TT_ArgumentType_Parameter: I32EnumAttrCase<"Parameter", 1, "parameter">;
def TT_ArgumentType_Constant : I32EnumAttrCase<"Constant", 2, "constant">;

def TT_ArgumentType : I32EnumAttr<"ArgumentType", "Argument Type",
                            [
                              TT_ArgumentType_Input,
                              TT_ArgumentType_Parameter,
                              TT_ArgumentType_Constant,
                            ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt";
  let stringToSymbolFnName = "ArgumentTypeStringToEnum";
  let symbolToStringFnName = "ArgumentTypeEnumToString";
}

#endif
