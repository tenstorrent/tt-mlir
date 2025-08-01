// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_SFPU_IR_SFPUOPS_TD
#define TTMLIR_DIALECT_SFPU_IR_SFPUOPS_TD

include "ttmlir/Dialect/SFPU/IR/SFPUOpsTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

//===----------------------------------------------------------------------===//
// Data Movement Operations
//===----------------------------------------------------------------------===//

def SFPU_MovOp : SFPU_UnaryOp<"mov", [Pure]> {
  let summary = "SFPU vector move/copy operation";
  let description = [{
    Maps to sfpmov instruction. Copies vector data between registers.
    
    Example:
    ```mlir
    %result = sfpu.mov %src : vector<64xf32>
    ```
  }];
  let arguments = (ins SFPU_VectorType:$src);
  let results = (outs SFPU_VectorType:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def SFPU_MovLvOp : SFPU_UnaryOp<"mov_lv", [Pure]> {
  let summary = "SFPU vector move operation (live variant)";
  let description = [{
    Maps to sfpmov_lv instruction. Live variant of vector move operation.
  }];
  let arguments = (ins SFPU_VectorType:$src);
  let results = (outs SFPU_VectorType:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Arithmetic Operations
//===----------------------------------------------------------------------===//

def SFPU_AddOp : SFPU_BinaryOp<"add", [Pure, Commutative]> {
  let summary = "SFPU vector addition";
  let description = [{
    Maps to sfpadd instruction. Performs element-wise addition of two SFPU vectors.
    
    Example:
    ```mlir
    %result = sfpu.add %lhs, %rhs : vector<64xf32>
    ```
  }];
  let arguments = (ins SFPU_VectorType:$lhs, SFPU_VectorType:$rhs);
  let results = (outs SFPU_VectorType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SFPU_AddLvOp : SFPU_BinaryOp<"add_lv", [Pure, Commutative]> {
  let summary = "SFPU vector addition (live variant)";
  let description = [{
    Maps to sfpadd_lv instruction. Live variant of vector addition.
  }];
  let arguments = (ins SFPU_VectorType:$lhs, SFPU_VectorType:$rhs);
  let results = (outs SFPU_VectorType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SFPU_MulOp : SFPU_BinaryOp<"mul", [Pure, Commutative]> {
  let summary = "SFPU vector multiplication";
  let description = [{
    Maps to sfpmul instruction. Performs element-wise multiplication of two SFPU vectors.
    
    Example:
    ```mlir
    %result = sfpu.mul %lhs, %rhs : vector<64xf32>
    ```
  }];
  let arguments = (ins SFPU_VectorType:$lhs, SFPU_VectorType:$rhs);
  let results = (outs SFPU_VectorType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SFPU_MulLvOp : SFPU_BinaryOp<"mul_lv", [Pure, Commutative]> {
  let summary = "SFPU vector multiplication (live variant)";
  let description = [{
    Maps to sfpmul_lv instruction. Live variant of vector multiplication.
  }];
  let arguments = (ins SFPU_VectorType:$lhs, SFPU_VectorType:$rhs);
  let results = (outs SFPU_VectorType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SFPU_MadOp : SFPU_TernaryOp<"mad", [Pure]> {
  let summary = "SFPU multiply-accumulate";
  let description = [{
    Maps to sfpmad instruction. Computes a * b + c for SFPU vectors.
    
    Example:
    ```mlir
    %result = sfpu.mad %a, %b, %c : vector<64xf32>
    ```
  }];
  let arguments = (ins SFPU_VectorType:$a, SFPU_VectorType:$b, SFPU_VectorType:$c);
  let results = (outs SFPU_VectorType:$result);
  let assemblyFormat = "$a `,` $b `,` $c attr-dict `:` type($a) `,` type($b) `,` type($c) `->` type($result)";
}

def SFPU_MadLvOp : SFPU_TernaryOp<"mad_lv", [Pure]> {
  let summary = "SFPU multiply-accumulate (live variant)";
  let description = [{
    Maps to sfpmad_lv instruction. Live variant of multiply-accumulate.
  }];
  let arguments = (ins SFPU_VectorType:$a, SFPU_VectorType:$b, SFPU_VectorType:$c);
  let results = (outs SFPU_VectorType:$result);
  let assemblyFormat = "$a `,` $b `,` $c attr-dict `:` type($a) `,` type($b) `,` type($c) `->` type($result)";
}

def SFPU_DivP2Op : SFPU_UnaryOp<"divp2", [Pure]> {
  let summary = "SFPU divide by power of 2";
  let description = [{
    Maps to sfpdivp2 instruction. Divides SFPU vector by 2^exp.
    
    Example:
    ```mlir
    %result = sfpu.divp2 %src, %exp : vector<64xf32>
    ```
  }];
  let arguments = (ins SFPU_VectorType:$src, I32:$exp);
  let results = (outs SFPU_VectorType:$result);
  let assemblyFormat = "$src `,` $exp attr-dict `:` type($src) `,` type($exp) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Bitwise Operations
//===----------------------------------------------------------------------===//

def SFPU_AndOp : SFPU_BinaryOp<"and", [Pure, Commutative]> {
  let summary = "SFPU bitwise AND";
  let description = [{
    Maps to sfpand instruction. Performs bitwise AND of two SFPU vectors.
  }];
  let arguments = (ins SFPU_VectorType:$lhs, SFPU_VectorType:$rhs);
  let results = (outs SFPU_VectorType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SFPU_OrOp : SFPU_BinaryOp<"or", [Pure, Commutative]> {
  let summary = "SFPU bitwise OR";
  let description = [{
    Maps to sfpor instruction. Performs bitwise OR of two SFPU vectors.
  }];
  let arguments = (ins SFPU_VectorType:$lhs, SFPU_VectorType:$rhs);
  let results = (outs SFPU_VectorType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SFPU_XorOp : SFPU_BinaryOp<"xor", [Pure, Commutative]> {
  let summary = "SFPU bitwise XOR";
  let description = [{
    Maps to sfpxor instruction. Performs bitwise XOR of two SFPU vectors.
  }];
  let arguments = (ins SFPU_VectorType:$lhs, SFPU_VectorType:$rhs);
  let results = (outs SFPU_VectorType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SFPU_NotOp : SFPU_UnaryOp<"not", [Pure]> {
  let summary = "SFPU bitwise NOT";
  let description = [{
    Maps to sfpnot instruction. Performs bitwise NOT of SFPU vector.
  }];
  let arguments = (ins SFPU_VectorType:$src);
  let results = (outs SFPU_VectorType:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Floating-Point Manipulation Operations
//===----------------------------------------------------------------------===//

def SFPU_SetExpOp : SFPU_BinaryOp<"setexp", [Pure]> {
  let summary = "Set exponent of SFPU vector";
  let description = [{
    Maps to sfpsetexp instruction. Sets the exponent bits of floating-point values.
  }];
  let arguments = (ins SFPU_VFloat:$src, SFPU_VInt:$exp);
  let results = (outs SFPU_VFloat:$result);
  let assemblyFormat = "$src `,` $exp attr-dict `:` type($src) `,` type($exp) `->` type($result)";
}

def SFPU_SetManOp : SFPU_BinaryOp<"setman", [Pure]> {
  let summary = "Set mantissa of SFPU vector";
  let description = [{
    Maps to sfpsetman instruction. Sets the mantissa bits of floating-point values.
  }];
  let arguments = (ins SFPU_VFloat:$src, SFPU_VInt:$man);
  let results = (outs SFPU_VFloat:$result);
  let assemblyFormat = "$src `,` $man attr-dict `:` type($src) `,` type($man) `->` type($result)";
}

def SFPU_SetSgnOp : SFPU_BinaryOp<"setsgn", [Pure]> {
  let summary = "Set sign of SFPU vector";
  let description = [{
    Maps to sfpsetsgn instruction. Sets the sign bits of floating-point values.
  }];
  let arguments = (ins SFPU_VFloat:$src, SFPU_VInt:$sgn);
  let results = (outs SFPU_VFloat:$result);
  let assemblyFormat = "$src `,` $sgn attr-dict `:` type($src) `,` type($sgn) `->` type($result)";
}

def SFPU_ExExpOp : SFPU_UnaryOp<"exexp", [Pure]> {
  let summary = "Extract exponent from SFPU vector";
  let description = [{
    Maps to sfpexexp instruction. Extracts the exponent bits from floating-point values.
  }];
  let arguments = (ins SFPU_VFloat:$src);
  let results = (outs SFPU_VInt:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def SFPU_ExManOp : SFPU_UnaryOp<"exman", [Pure]> {
  let summary = "Extract mantissa from SFPU vector";
  let description = [{
    Maps to sfpexman instruction. Extracts the mantissa bits from floating-point values.
  }];
  let arguments = (ins SFPU_VFloat:$src);
  let results = (outs SFPU_VInt:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def SFPU_AbsOp : SFPU_UnaryOp<"abs", [Pure]> {
  let summary = "SFPU absolute value";
  let description = [{
    Maps to sfpabs instruction. Computes absolute value of SFPU vector elements.
  }];
  let arguments = (ins SFPU_VFloat:$src);
  let results = (outs SFPU_VFloat:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def SFPU_LzOp : SFPU_UnaryOp<"lz", [Pure]> {
  let summary = "SFPU leading zero count";
  let description = [{
    Maps to sfplz instruction. Counts leading zeros in SFPU vector elements.
  }];
  let arguments = (ins SFPU_VFloat:$src);
  let results = (outs SFPU_VInt:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Comparison Operations
//===----------------------------------------------------------------------===//

def SFPU_XfCmpSOp : SFPU_ComparisonOp<"xfcmps", [Pure]> {
  let summary = "SFPU scalar floating-point comparison";
  let description = [{
    Maps to sfpxfcmps instruction. Performs scalar floating-point comparison.
  }];
  let arguments = (ins SFPU_VFloat:$lhs, SFPU_VFloat:$rhs, SFPU_CmpPredicateAttr:$predicate);
  let results = (outs I1:$result);
  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SFPU_XfCmpVOp : SFPU_ComparisonOp<"xfcmpv", [Pure]> {
  let summary = "SFPU vector floating-point comparison";
  let description = [{
    Maps to sfpxfcmpv instruction. Performs vector floating-point comparison.
  }];
  let arguments = (ins SFPU_VFloat:$lhs, SFPU_VFloat:$rhs, SFPU_CmpPredicateAttr:$predicate);
  let results = (outs SFPU_VInt:$result);
  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SFPU_GtOp : SFPU_ComparisonOp<"gt", [Pure]> {
  let summary = "SFPU greater than comparison";
  let description = [{
    Maps to sfpgt instruction. Performs greater than comparison.
  }];
  let arguments = (ins SFPU_VFloat:$lhs, SFPU_VFloat:$rhs);
  let results = (outs SFPU_VInt:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SFPU_LeOp : SFPU_ComparisonOp<"le", [Pure]> {
  let summary = "SFPU less than or equal comparison";
  let description = [{
    Maps to sfple instruction. Performs less than or equal comparison.
  }];
  let arguments = (ins SFPU_VFloat:$lhs, SFPU_VFloat:$rhs);
  let results = (outs SFPU_VInt:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Type Conversion Operations
//===----------------------------------------------------------------------===//

def SFPU_CastOp : SFPU_ConversionOp<"cast", [Pure]> {
  let summary = "SFPU type conversion";
  let description = [{
    Maps to sfpcast instruction. Converts between different SFPU vector types.
  }];
  let arguments = (ins SFPU_VectorType:$src);
  let results = (outs SFPU_VectorType:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Specialized Operations
//===----------------------------------------------------------------------===//

def SFPU_ShftOp : SFPU_UnaryOp<"shft", [Pure]> {
  let summary = "SFPU shift operation";
  let description = [{
    Maps to sfpshft instruction. Performs shift operation on SFPU vector.
  }];
  let arguments = (ins SFPU_VectorType:$src, I32:$amount);
  let results = (outs SFPU_VectorType:$result);
  let assemblyFormat = "$src `,` $amount attr-dict `:` type($src) `,` type($amount) `->` type($result)";
}

def SFPU_StochRndOp : SFPU_UnaryOp<"stochrnd", [Pure]> {
  let summary = "SFPU stochastic rounding";
  let description = [{
    Maps to sfpstochrnd instruction. Performs stochastic rounding on SFPU vector.
  }];
  let arguments = (ins SFPU_VFloat:$src);
  let results = (outs SFPU_VFloat:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def SFPU_LutOp : SFPU_UnaryOp<"lut", [Pure]> {
  let summary = "SFPU lookup table operation";
  let description = [{
    Maps to sfplut instruction. Performs lookup table operation on SFPU vector.
  }];
  let arguments = (ins SFPU_VFloat:$src, ArrayAttr:$table);
  let results = (outs SFPU_VFloat:$result);
  let assemblyFormat = "$src `,` $table attr-dict `:` type($src) `->` type($result)";
}

def SFPU_ARecipOp : SFPU_UnaryOp<"arecip", [Pure]> {
  let summary = "SFPU reciprocal approximation";
  let description = [{
    Maps to sfparecip instruction. Computes reciprocal approximation of SFPU vector.
  }];
  let arguments = (ins SFPU_VFloat:$src);
  let results = (outs SFPU_VFloat:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

#endif // TTMLIR_DIALECT_SFPU_IR_SFPUOPS_TD