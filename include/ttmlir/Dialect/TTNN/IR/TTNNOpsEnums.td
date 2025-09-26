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

def TTNN_TensorMemoryLayout_Interleaved : I32EnumAttrCase<"Interleaved", 0, "interleaved">;
def TTNN_TensorMemoryLayout_HeightSharded : I32EnumAttrCase<"HeightSharded", 2, "height_sharded">;
def TTNN_TensorMemoryLayout_WidthSharded : I32EnumAttrCase<"WidthSharded", 3, "width_sharded">;
def TTNN_TensorMemoryLayout_BlockSharded : I32EnumAttrCase<"BlockSharded", 4, "block_sharded">;

def TTNN_TensorMemoryLayout : I32EnumAttr<"TensorMemoryLayout", "TTNN Tensor Memory Layout",
                           [
                            TTNN_TensorMemoryLayout_Interleaved,
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

def TTNN_UnaryOpType_Exp  : I32EnumAttrCase<"Exp", 0, "exp">;
def TTNN_UnaryOpType_Recip  : I32EnumAttrCase<"Recip", 1, "recip">;
def TTNN_UnaryOpType_Gelu  : I32EnumAttrCase<"Gelu", 2, "gelu">;
def TTNN_UnaryOpType_Relu  : I32EnumAttrCase<"Relu", 3, "relu">;
def TTNN_UnaryOpType_Sqrt  : I32EnumAttrCase<"Sqrt", 4, "sqrt">;
def TTNN_UnaryOpType_Sigmoid  : I32EnumAttrCase<"Sigmoid", 5, "sigmoid">;
def TTNN_UnaryOpType_Log  : I32EnumAttrCase<"Log", 6, "log">;
def TTNN_UnaryOpType_Tanh  : I32EnumAttrCase<"Tanh", 7, "tanh">;
def TTNN_UnaryOpType_Log2  : I32EnumAttrCase<"Log2", 8, "log2">;
def TTNN_UnaryOpType_Log10  : I32EnumAttrCase<"Log10", 9, "log10">;
def TTNN_UnaryOpType_Sin  : I32EnumAttrCase<"Sin", 10, "sin">;
def TTNN_UnaryOpType_Cos  : I32EnumAttrCase<"Cos", 11, "cos">;
def TTNN_UnaryOpType_Abs  : I32EnumAttrCase<"Abs", 12, "abs">;
def TTNN_UnaryOpType_AbsInt32  : I32EnumAttrCase<"AbsInt32", 13, "abs_int32">;
def TTNN_UnaryOpType_Sign  : I32EnumAttrCase<"Sign", 14, "sign">;
def TTNN_UnaryOpType_Square  : I32EnumAttrCase<"Square", 15, "square">;
def TTNN_UnaryOpType_Eqz  : I32EnumAttrCase<"Eqz", 16, "eqz">;
def TTNN_UnaryOpType_Nez  : I32EnumAttrCase<"Nez", 17, "nez">;
def TTNN_UnaryOpType_Gtz  : I32EnumAttrCase<"Gtz", 18, "gtz">;
def TTNN_UnaryOpType_Ltz  : I32EnumAttrCase<"Ltz", 19, "ltz">;
def TTNN_UnaryOpType_Gez  : I32EnumAttrCase<"Gez", 20, "gez">;
def TTNN_UnaryOpType_Lez  : I32EnumAttrCase<"Lez", 21, "Lez">;
def TTNN_UnaryOpType_ReluMax  : I32EnumAttrCase<"ReluMax", 22, "relu_max">;
def TTNN_UnaryOpType_ReluMin  : I32EnumAttrCase<"ReluMin", 23, "relu_min">;
def TTNN_UnaryOpType_Power  : I32EnumAttrCase<"Power", 24, "power">;
def TTNN_UnaryOpType_LeakyRelu  : I32EnumAttrCase<"LeakyRelu", 25, "leaky_relu">;
def TTNN_UnaryOpType_Elu  : I32EnumAttrCase<"Elu", 26, "elu">;
def TTNN_UnaryOpType_Exp2  : I32EnumAttrCase<"Exp2", 27, "exp2">;
def TTNN_UnaryOpType_Heaviside  : I32EnumAttrCase<"Heaviside", 28, "heaviside">;
def TTNN_UnaryOpType_Expm1  : I32EnumAttrCase<"Expm1", 29, "expm1">;
def TTNN_UnaryOpType_Signbit  : I32EnumAttrCase<"Signbit", 30, "signbit">;
def TTNN_UnaryOpType_Asin  : I32EnumAttrCase<"Asin", 31, "asin">;
def TTNN_UnaryOpType_Acos  : I32EnumAttrCase<"Acos", 32, "acos">;
def TTNN_UnaryOpType_Rsqrt  : I32EnumAttrCase<"Rsqrt", 33, "rsqrt">;
def TTNN_UnaryOpType_Relu6  : I32EnumAttrCase<"Relu6", 34, "relu6">;
def TTNN_UnaryOpType_Atan  : I32EnumAttrCase<"Atan", 35, "atan">;
def TTNN_UnaryOpType_Erf  : I32EnumAttrCase<"Erf", 36, "erf">;
def TTNN_UnaryOpType_Erfc  : I32EnumAttrCase<"Erfc", 37, "erfc">;
def TTNN_UnaryOpType_IsInf  : I32EnumAttrCase<"IsInf", 38, "is_inf">;
def TTNN_UnaryOpType_IsPosInf  : I32EnumAttrCase<"IsPosInf", 39, "is_pos_inf">;
def TTNN_UnaryOpType_IsNegInf  : I32EnumAttrCase<"IsNegInf", 40, "is_neg_inf">;
def TTNN_UnaryOpType_IsNan  : I32EnumAttrCase<"IsNan", 41, "is_nan">;
def TTNN_UnaryOpType_LogicalNotUnary  : I32EnumAttrCase<"LogicalNotUnary", 42, "logical_not_unary">;
def TTNN_UnaryOpType_IsFinite  : I32EnumAttrCase<"IsFinite", 43, "is_finite">;
def TTNN_UnaryOpType_Erfinv  : I32EnumAttrCase<"Erfinv", 44, "erfinv">;
def TTNN_UnaryOpType_I0  : I32EnumAttrCase<"I0", 45, "i0">;
def TTNN_UnaryOpType_I1  : I32EnumAttrCase<"I1", 46, "i1">;
def TTNN_UnaryOpType_Tan  : I32EnumAttrCase<"Tan", 47, "tan">;
def TTNN_UnaryOpType_Rsub  : I32EnumAttrCase<"Rsub", 48, "rsub">;
def TTNN_UnaryOpType_Rdiv  : I32EnumAttrCase<"Rdiv", 49, "rdiv">;
def TTNN_UnaryOpType_Silu  : I32EnumAttrCase<"Silu", 50, "silu">;
def TTNN_UnaryOpType_SoftPlus  : I32EnumAttrCase<"SoftPlus", 51, "soft_plus">;
def TTNN_UnaryOpType_Identity  : I32EnumAttrCase<"Identity", 52, "identity">;
def TTNN_UnaryOpType_Neg  : I32EnumAttrCase<"Neg", 53, "neg">;
def TTNN_UnaryOpType_AddUnarySfpu  : I32EnumAttrCase<"AddUnarySfpu", 54, "add_unary_sfpu">;
def TTNN_UnaryOpType_SubUnarySfpu  : I32EnumAttrCase<"SubUnarySfpu", 55, "sub_unary_sfpu">;
def TTNN_UnaryOpType_MulUnarySfpu  : I32EnumAttrCase<"MulUnarySfpu", 56, "mul_unary_sfpu">;
def TTNN_UnaryOpType_DivUnarySfpu  : I32EnumAttrCase<"DivUnarySfpu", 57, "div_unary_sfpu">;
def TTNN_UnaryOpType_IdentityUint32  : I32EnumAttrCase<"IdentityUint32", 58, "identity_uint32">;
def TTNN_UnaryOpType_UnaryNe  : I32EnumAttrCase<"UnaryNe", 59, "unary_ne">;
def TTNN_UnaryOpType_UnaryGt  : I32EnumAttrCase<"UnaryGt", 60, "unary_gt">;
def TTNN_UnaryOpType_UnaryLt  : I32EnumAttrCase<"UnaryLt", 61, "unary_lt">;
def TTNN_UnaryOpType_TiledProd  : I32EnumAttrCase<"TiledProd", 62, "tiled_prod">;
def TTNN_UnaryOpType_Typecast  : I32EnumAttrCase<"Typecast", 63, "typecast">;
def TTNN_UnaryOpType_BitwiseXor  : I32EnumAttrCase<"BitwiseXor", 64, "bitwise_xor">;
def TTNN_UnaryOpType_BitwiseNot  : I32EnumAttrCase<"BitwiseNot", 65, "bitwise_not">;
def TTNN_UnaryOpType_BitwiseAnd  : I32EnumAttrCase<"BitwiseAnd", 66, "bitwise_and">;
def TTNN_UnaryOpType_BitwiseOr  : I32EnumAttrCase<"BitwiseOr", 67, "bitwise_or">;
def TTNN_UnaryOpType_RightShift  : I32EnumAttrCase<"RightShift", 68, "right_shift">;
def TTNN_UnaryOpType_Floor  : I32EnumAttrCase<"Floor", 69, "floor">;
def TTNN_UnaryOpType_Ceil  : I32EnumAttrCase<"Ceil", 70, "ceil">;
def TTNN_UnaryOpType_Round  : I32EnumAttrCase<"Round", 71, "round">;
def TTNN_UnaryOpType_LeftShift  : I32EnumAttrCase<"LeftShift", 72, "left_shift">;
def TTNN_UnaryOpType_Remainder  : I32EnumAttrCase<"Remainder", 73, "remainder">;
def TTNN_UnaryOpType_Fmod  : I32EnumAttrCase<"Fmod", 74, "fmod">;
def TTNN_UnaryOpType_Dropout  : I32EnumAttrCase<"Dropout", 75, "dropout">;
def TTNN_UnaryOpType_Fill  : I32EnumAttrCase<"Fill", 76, "fill">;
def TTNN_UnaryOpType_PreluSfpu  : I32EnumAttrCase<"PreluSfpu", 77, "prelu_sfpu">;
def TTNN_UnaryOpType_ZeroPoint  : I32EnumAttrCase<"ZeroPoint", 78, "zero_point">;

def TTNN_UnaryOpType : I32EnumAttr<"UnaryOpType", "TTNN Unary Operation Type",
                          [
                            TTNN_UnaryOpType_Exp,
                            TTNN_UnaryOpType_Recip,
                            TTNN_UnaryOpType_Gelu,
                            TTNN_UnaryOpType_Relu,
                            TTNN_UnaryOpType_Sqrt,
                            TTNN_UnaryOpType_Sigmoid,
                            TTNN_UnaryOpType_Log,
                            TTNN_UnaryOpType_Tanh,
                            TTNN_UnaryOpType_Log2,
                            TTNN_UnaryOpType_Log10,
                            TTNN_UnaryOpType_Sin,
                            TTNN_UnaryOpType_Cos,
                            TTNN_UnaryOpType_Abs,
                            TTNN_UnaryOpType_AbsInt32,
                            TTNN_UnaryOpType_Sign,
                            TTNN_UnaryOpType_Square,
                            TTNN_UnaryOpType_Eqz,
                            TTNN_UnaryOpType_Nez,
                            TTNN_UnaryOpType_Gtz,
                            TTNN_UnaryOpType_Ltz,
                            TTNN_UnaryOpType_Gez,
                            TTNN_UnaryOpType_Lez,
                            TTNN_UnaryOpType_ReluMax,
                            TTNN_UnaryOpType_ReluMin,
                            TTNN_UnaryOpType_Power,
                            TTNN_UnaryOpType_LeakyRelu,
                            TTNN_UnaryOpType_Elu,
                            TTNN_UnaryOpType_Exp2,
                            TTNN_UnaryOpType_Heaviside,
                            TTNN_UnaryOpType_Expm1,
                            TTNN_UnaryOpType_Signbit,
                            TTNN_UnaryOpType_Asin,
                            TTNN_UnaryOpType_Acos,
                            TTNN_UnaryOpType_Rsqrt,
                            TTNN_UnaryOpType_Relu6,
                            TTNN_UnaryOpType_Atan,
                            TTNN_UnaryOpType_Erf,
                            TTNN_UnaryOpType_Erfc,
                            TTNN_UnaryOpType_IsInf,
                            TTNN_UnaryOpType_IsPosInf,
                            TTNN_UnaryOpType_IsNegInf,
                            TTNN_UnaryOpType_IsNan,
                            TTNN_UnaryOpType_LogicalNotUnary,
                            TTNN_UnaryOpType_IsFinite,
                            TTNN_UnaryOpType_Erfinv,
                            TTNN_UnaryOpType_I0,
                            TTNN_UnaryOpType_I1,
                            TTNN_UnaryOpType_Tan,
                            TTNN_UnaryOpType_Rsub,
                            TTNN_UnaryOpType_Rdiv,
                            TTNN_UnaryOpType_Silu,
                            TTNN_UnaryOpType_SoftPlus,
                            TTNN_UnaryOpType_Identity,
                            TTNN_UnaryOpType_Neg,
                            TTNN_UnaryOpType_AddUnarySfpu,
                            TTNN_UnaryOpType_SubUnarySfpu,
                            TTNN_UnaryOpType_MulUnarySfpu,
                            TTNN_UnaryOpType_DivUnarySfpu,
                            TTNN_UnaryOpType_IdentityUint32,
                            TTNN_UnaryOpType_UnaryNe,
                            TTNN_UnaryOpType_UnaryGt,
                            TTNN_UnaryOpType_UnaryLt,
                            TTNN_UnaryOpType_TiledProd,
                            TTNN_UnaryOpType_Typecast,
                            TTNN_UnaryOpType_BitwiseXor,
                            TTNN_UnaryOpType_BitwiseNot,
                            TTNN_UnaryOpType_BitwiseAnd,
                            TTNN_UnaryOpType_BitwiseOr,
                            TTNN_UnaryOpType_RightShift,
                            TTNN_UnaryOpType_Floor,
                            TTNN_UnaryOpType_Ceil,
                            TTNN_UnaryOpType_Round,
                            TTNN_UnaryOpType_LeftShift,
                            TTNN_UnaryOpType_Remainder,
                            TTNN_UnaryOpType_Fmod,
                            TTNN_UnaryOpType_Dropout,
                            TTNN_UnaryOpType_Fill,
                            TTNN_UnaryOpType_PreluSfpu,
                            TTNN_UnaryOpType_ZeroPoint
                          ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttnn";
}

def TTNN_ShardOrientation_RowMajor : I32EnumAttrCase<"RowMajor", 0, "row_major">;
def TTNN_ShardOrientation_ColMajor : I32EnumAttrCase<"ColMajor", 1, "col_major">;

def TTNN_ShardOrientation : I32EnumAttr<"ShardOrientation", "TTNN ShardOrientation",
                           [
                            TTNN_ShardOrientation_RowMajor,
                            TTNN_ShardOrientation_ColMajor
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttnn";
}

def TTNN_MathFidelityLoFi : I32EnumAttrCase<"LoFi", 0, "lofi">;
def TTNN_MathFidelityHiFi2 : I32EnumAttrCase<"HiFi2", 2, "hifi2">;
def TTNN_MathFidelityHiFi3 : I32EnumAttrCase<"HiFi3", 3, "hifi3">;
def TTNN_MathFidelityHiFi4 : I32EnumAttrCase<"HiFi4", 4, "hifi4">;

def TTNN_MathFidelity : I32EnumAttr<"MathFidelity", "TT MathFidelity",
                           [
                            TTNN_MathFidelityLoFi,
                            TTNN_MathFidelityHiFi2,
                            TTNN_MathFidelityHiFi3,
                            TTNN_MathFidelityHiFi4,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttnn";
}

def TTNN_ComputeKernelMathFidelityLoFi : I32EnumAttrCase<"LoFi", 0, "lofi">;
def TTNN_ComputeKernelMathFidelityHiFi2 : I32EnumAttrCase<"HiFi2", 2, "hifi2">;
def TTNN_ComputeKernelMathFidelityHiFi3 : I32EnumAttrCase<"HiFi3", 3, "hifi3">;
def TTNN_ComputeKernelMathFidelityHiFi4 : I32EnumAttrCase<"HiFi4", 4, "hifi4">;

def TTNN_ComputeKernelMathFidelity : I32EnumAttr<"ComputeKernelMathFidelity", "TT ComputeKernelMathFidelity",
                           [
                            TTNN_ComputeKernelMathFidelityLoFi,
                            TTNN_ComputeKernelMathFidelityHiFi2,
                            TTNN_ComputeKernelMathFidelityHiFi3,
                            TTNN_ComputeKernelMathFidelityHiFi4,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttnn";
}

def TTNN_ComputeKernelUnpackToDestFp32 : I32EnumAttrCase<"Fp32", 0, "fp32">;
def TTNN_ComputeKernelUnpackToDestDefault : I32EnumAttrCase<"Default", 1, "default">;

def TTNN_ComputeKernelUnpackToDestMode : I32EnumAttr<"ComputeKernelUnpackToDestMode", "TT ComputeKernelUnpackToDestMode",
                           [
                            TTNN_ComputeKernelUnpackToDestFp32,
                            TTNN_ComputeKernelUnpackToDestDefault,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttnn";
}

def TTNN_KernelCoreTypeWorker : I32EnumAttrCase<"Worker", 0, "worker">;
def TTNN_KernelCoreTypeEth : I32EnumAttrCase<"Eth", 1, "eth">;

def TTNN_KernelCoreType : I32EnumAttr<"KernelCoreType", "TT KernelCoreType",
                           [
                            TTNN_KernelCoreTypeWorker,
                            TTNN_KernelCoreTypeEth,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttnn";
}

#endif
