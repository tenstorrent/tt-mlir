// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_SFPI_IR_SFPIOPS_TD
#define TTMLIR_DIALECT_SFPI_IR_SFPIOPS_TD

include "ttmlir/Dialect/SFPI/IR/SFPIOpsTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

//===----------------------------------------------------------------------===//
// Data Movement Operations
//===----------------------------------------------------------------------===//

def SFPI_MovOp : SFPI_UnaryOp<"mov", [Pure]> {
  let summary = "SFPI vector move/copy operation";
  let description = [{
    Maps to sfpmov instruction. Copies vector data between registers.
    
    Example:
    ```mlir
    %result = sfpi.mov %src : vector<4x8xf32>
    ```
  }];
  let arguments = (ins SFPI_VectorType:$src);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def SFPI_MovLvOp : SFPI_UnaryOp<"mov_lv", [Pure]> {
  let summary = "SFPI vector move operation (live variant)";
  let description = [{
    Maps to sfpmov_lv instruction. Live variant of vector move operation.
  }];
  let arguments = (ins SFPI_VectorType:$src);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Arithmetic Operations
//===----------------------------------------------------------------------===//

def SFPI_AddOp : SFPI_BinaryOp<"add", [Pure, Commutative]> {
  let summary = "SFPI vector addition";
  let description = [{
    Maps to sfpadd instruction. Performs element-wise addition of two SFPI vectors.
    
    Example:
    ```mlir
    %result = sfpi.add %lhs, %rhs : vector<4x8xf32>
    ```
  }];
  let arguments = (ins SFPI_VectorType:$lhs, SFPI_VectorType:$rhs);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SFPI_AddLvOp : SFPI_BinaryOp<"add_lv", [Pure, Commutative]> {
  let summary = "SFPI vector addition (live variant)";
  let description = [{
    Maps to sfpadd_lv instruction. Live variant of vector addition.
  }];
  let arguments = (ins SFPI_VectorType:$lhs, SFPI_VectorType:$rhs);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SFPI_MulOp : SFPI_BinaryOp<"mul", [Pure, Commutative]> {
  let summary = "SFPI vector multiplication";
  let description = [{
    Maps to sfpmul instruction. Performs element-wise multiplication of two SFPI vectors.
    
    Example:
    ```mlir
    %result = sfpi.mul %lhs, %rhs : vector<4x8xf32>
    ```
  }];
  let arguments = (ins SFPI_VectorType:$lhs, SFPI_VectorType:$rhs);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SFPI_MulLvOp : SFPI_BinaryOp<"mul_lv", [Pure, Commutative]> {
  let summary = "SFPI vector multiplication (live variant)";
  let description = [{
    Maps to sfpmul_lv instruction. Live variant of vector multiplication.
  }];
  let arguments = (ins SFPI_VectorType:$lhs, SFPI_VectorType:$rhs);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SFPI_MadOp : SFPI_TernaryOp<"mad"> {
  let summary = "SFPI multiply-accumulate";
  let description = [{
    Maps to sfpmad instruction. Computes a * b + c for SFPI vectors.
    
    Example:
    ```mlir
    %result = sfpi.mad %a, %b, %c : vector<4x8xf32>
    ```
  }];
  let arguments = (ins SFPI_VectorType:$a, SFPI_VectorType:$b, SFPI_VectorType:$c);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$a `,` $b `,` $c attr-dict `:` type($a) `,` type($b) `,` type($c) `->` type($result)";
}

def SFPI_MadLvOp : SFPI_TernaryOp<"mad_lv"> {
  let summary = "SFPI multiply-accumulate (live variant)";
  let description = [{
    Maps to sfpmad_lv instruction. Live variant of multiply-accumulate.
  }];
  let arguments = (ins SFPI_VectorType:$a, SFPI_VectorType:$b, SFPI_VectorType:$c);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$a `,` $b `,` $c attr-dict `:` type($a) `,` type($b) `,` type($c) `->` type($result)";
}

def SFPI_DivP2Op : SFPI_UnaryOp<"divp2"> {
  let summary = "SFPI divide by power of 2";
  let description = [{
    Maps to sfpdivp2 instruction. Divides SFPI vector by 2^exp.
    
    Example:
    ```mlir
    %result = sfpi.divp2 %src, %exp : vector<4x8xf32>
    ```
  }];
  let arguments = (ins SFPI_VectorType:$src, I32:$exp);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$src `,` $exp attr-dict `:` type($src) `,` type($exp) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Bitwise Operations
//===----------------------------------------------------------------------===//

def SFPI_AndOp : SFPI_BinaryOp<"and", [Pure, Commutative]> {
  let summary = "SFPI bitwise AND";
  let description = [{
    Maps to sfpand instruction. Performs bitwise AND of two SFPI vectors.
  }];
  let arguments = (ins SFPI_VectorType:$lhs, SFPI_VectorType:$rhs);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SFPI_OrOp : SFPI_BinaryOp<"or", [Pure, Commutative]> {
  let summary = "SFPI bitwise OR";
  let description = [{
    Maps to sfpor instruction. Performs bitwise OR of two SFPI vectors.
  }];
  let arguments = (ins SFPI_VectorType:$lhs, SFPI_VectorType:$rhs);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SFPI_XorOp : SFPI_BinaryOp<"xor", [Pure, Commutative]> {
  let summary = "SFPI bitwise XOR";
  let description = [{
    Maps to sfpxor instruction. Performs bitwise XOR of two SFPI vectors.
  }];
  let arguments = (ins SFPI_VectorType:$lhs, SFPI_VectorType:$rhs);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SFPI_NotOp : SFPI_UnaryOp<"not"> {
  let summary = "SFPI bitwise NOT";
  let description = [{
    Maps to sfpnot instruction. Performs bitwise NOT of SFPI vector.
  }];
  let arguments = (ins SFPI_VectorType:$src);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Floating-Point Manipulation Operations
//===----------------------------------------------------------------------===//

def SFPI_SetExpOp : SFPI_BinaryOp<"setexp"> {
  let summary = "Set exponent of SFPI vector";
  let description = [{
    Maps to sfpsetexp instruction. Sets the exponent bits of floating-point values.
  }];
  let arguments = (ins SFPI_VFloat:$src, SFPI_VInt:$exp);
  let results = (outs SFPI_VFloat:$result);
  let assemblyFormat = "$src `,` $exp attr-dict `:` type($src) `,` type($exp) `->` type($result)";
}

def SFPI_SetManOp : SFPI_BinaryOp<"setman"> {
  let summary = "Set mantissa of SFPI vector";
  let description = [{
    Maps to sfpsetman instruction. Sets the mantissa bits of floating-point values.
  }];
  let arguments = (ins SFPI_VFloat:$src, SFPI_VInt:$man);
  let results = (outs SFPI_VFloat:$result);
  let assemblyFormat = "$src `,` $man attr-dict `:` type($src) `,` type($man) `->` type($result)";
}

def SFPI_SetSgnOp : SFPI_BinaryOp<"setsgn"> {
  let summary = "Set sign of SFPI vector";
  let description = [{
    Maps to sfpsetsgn instruction. Sets the sign bits of floating-point values.
  }];
  let arguments = (ins SFPI_VFloat:$src, SFPI_VInt:$sgn);
  let results = (outs SFPI_VFloat:$result);
  let assemblyFormat = "$src `,` $sgn attr-dict `:` type($src) `,` type($sgn) `->` type($result)";
}

def SFPI_ExExpOp : SFPI_UnaryOp<"exexp"> {
  let summary = "Extract exponent from SFPI vector";
  let description = [{
    Maps to sfpexexp instruction. Extracts the exponent bits from floating-point values.
  }];
  let arguments = (ins SFPI_VFloat:$src);
  let results = (outs SFPI_VInt:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def SFPI_ExManOp : SFPI_UnaryOp<"exman"> {
  let summary = "Extract mantissa from SFPI vector";
  let description = [{
    Maps to sfpexman instruction. Extracts the mantissa bits from floating-point values.
  }];
  let arguments = (ins SFPI_VFloat:$src);
  let results = (outs SFPI_VInt:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def SFPI_AbsOp : SFPI_UnaryOp<"abs"> {
  let summary = "SFPI absolute value";
  let description = [{
    Maps to sfpabs instruction. Computes absolute value of SFPI vector elements.
  }];
  let arguments = (ins SFPI_VFloat:$src);
  let results = (outs SFPI_VFloat:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def SFPI_LzOp : SFPI_UnaryOp<"lz"> {
  let summary = "SFPI leading zero count";
  let description = [{
    Maps to sfplz instruction. Counts leading zeros in SFPI vector elements.
  }];
  let arguments = (ins SFPI_VFloat:$src);
  let results = (outs SFPI_VInt:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Comparison Operations
//===----------------------------------------------------------------------===//

def SFPI_XfCmpSOp : SFPI_ComparisonOp<"xfcmps"> {
  let summary = "SFPI scalar floating-point comparison";
  let description = [{
    Maps to sfpxfcmps instruction. Performs scalar floating-point comparison.
  }];
  let arguments = (ins SFPI_VFloat:$lhs, SFPI_VFloat:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SFPI_XfCmpVOp : SFPI_ComparisonOp<"xfcmpv"> {
  let summary = "SFPI vector floating-point comparison";
  let description = [{
    Maps to sfpxfcmpv instruction. Performs vector floating-point comparison.
  }];
  let arguments = (ins SFPI_VFloat:$lhs, SFPI_VFloat:$rhs);
  let results = (outs SFPI_VInt:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SFPI_GtOp : SFPI_ComparisonOp<"gt"> {
  let summary = "SFPI greater than comparison";
  let description = [{
    Maps to sfpgt instruction. Performs greater than comparison.
  }];
  let arguments = (ins SFPI_VFloat:$lhs, SFPI_VFloat:$rhs);
  let results = (outs SFPI_VInt:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def SFPI_LeOp : SFPI_ComparisonOp<"le"> {
  let summary = "SFPI less than or equal comparison";
  let description = [{
    Maps to sfple instruction. Performs less than or equal comparison.
  }];
  let arguments = (ins SFPI_VFloat:$lhs, SFPI_VFloat:$rhs);
  let results = (outs SFPI_VInt:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Type Conversion Operations
//===----------------------------------------------------------------------===//

def SFPI_CastOp : SFPI_ConversionOp<"cast"> {
  let summary = "SFPI type conversion";
  let description = [{
    Maps to sfpcast instruction. Converts between different SFPI vector types.
  }];
  let arguments = (ins SFPI_VectorType:$src);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Specialized Operations
//===----------------------------------------------------------------------===//

def SFPI_ShftOp : SFPI_UnaryOp<"shft"> {
  let summary = "SFPI shift operation";
  let description = [{
    Maps to sfpshft instruction. Performs shift operation on SFPI vector.
  }];
  let arguments = (ins SFPI_VectorType:$src, I32:$amount);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$src `,` $amount attr-dict `:` type($src) `,` type($amount) `->` type($result)";
}

def SFPI_StochRndOp : SFPI_UnaryOp<"stochrnd"> {
  let summary = "SFPI stochastic rounding";
  let description = [{
    Maps to sfpstochrnd instruction. Performs stochastic rounding on SFPI vector.
  }];
  let arguments = (ins SFPI_VFloat:$src);
  let results = (outs SFPI_VFloat:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def SFPI_LutOp : SFPI_UnaryOp<"lut"> {
  let summary = "SFPI lookup table operation";
  let description = [{
    Maps to sfplut instruction. Performs lookup table operation on SFPI vector.
  }];
  let arguments = (ins SFPI_VFloat:$src);
  let results = (outs SFPI_VFloat:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def SFPI_ARecipOp : SFPI_UnaryOp<"arecip"> {
  let summary = "SFPI reciprocal approximation";
  let description = [{
    Maps to sfparecip instruction. Computes reciprocal approximation of SFPI vector.
  }];
  let arguments = (ins SFPI_VFloat:$src);
  let results = (outs SFPI_VFloat:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

#endif // TTMLIR_DIALECT_SFPI_IR_SFPIOPS_TD