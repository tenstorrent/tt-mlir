// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_SFPI_IR_SFPIOPS_TD
#define TTMLIR_DIALECT_SFPI_IR_SFPIOPS_TD

include "ttmlir/Dialect/SFPI/IR/SFPIOpsTypes.td"
include "ttmlir/Dialect/SFPI/IR/SFPIOpsAttrs.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

def IndexLike: AnyTypeOf<[Index, I32]>;

//===----------------------------------------------------------------------===//
// SFPI Operation Set
// Source:
// https://github.com/tenstorrent/sfpi/blob/main/include/wormhole/sfpi_hw.h
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Data Movement Operations
//===----------------------------------------------------------------------===//

def SFPI_LoadOp : SFPI_Op<"load"> {
  let summary = "Move from Dst to LReg";
  let description = [{
    Move (up to) 32 datums from the even or odd columns of four consecutive rows of Dst to an LReg. To bridge the gap between Dst data types and LReg data types, some data type conversions are supported as part of the instruction, though software might still want a subsequent SFPSTOCHRND or SFPCAST instruction to achieve a richer set of conversions.

    Example:
    ```mlir
    %result = sfpi.load %addr {format = #sfpi<load_format FP32>, addr_mode = #sfpi<load_addr_mode NOINC>} : !sfpi.ptr -> vector<4x8xf32>
    ```
  }];
  let arguments = (ins IndexLike:$addr, SFPI_LoadMod0Attr:$mod0,
      SFPI_LoadAddrModeAttr:$mode);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$addr attr-dict `:` type($addr) `->` type($result)";
}

def SFPI_StoreOp : SFPI_Op<"store"> {
  let summary = "Move from LReg to Dst";
  let description = [{
    Move (up to) 32 datums from an LReg to the even or odd columns of four consecutive rows of Dst. To bridge the gap between LReg data types and Dst data types, some data type conversions are supported as part of the instruction, though software might still want a preceding SFPSTOCHRND or SFPCAST instruction to achieve a richer set of conversions.

    Example:
    ```mlir
    sfpi.store %value, %addr {format = #sfpi<store_format FP32>, addr_mode = #sfpi<store_addr_mode NOINC>} : vector<4x8xf32>, !sfpi.ptr
    ```
  }];
  let arguments = (ins SFPI_VectorType:$value, IndexLike:$addr,
      SFPI_StoreMod0Attr:$mod0, SFPI_StoreAddrModeAttr:$mode);
  let results = (outs);
  let assemblyFormat =
      "$value `,` $addr attr-dict `:` type($value) `,` type($addr)";
}

def SFPI_XLoadIOp : SFPI_Op<"xloadi", [Pure]> {
  let summary = "SFPI extended load immediate operation";
  let description = [{
    Maps to sfpxloadi instruction. Extended load immediate value into SFPI vector register.

    Example:
    ```mlir
    %result = sfpi.xloadi %imm {mode = #sfpi<xload_imm_mode FLOAT>} : f32 -> vector<4x8xf32>
    ```
  }];
  let arguments = (ins I32:$imm, SFPI_XLoadIMod0Attr:$mod0);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$imm attr-dict `:` type($imm) `->` type($result)";
}

def SFPI_MovOp : SFPI_UnaryOp<"mov", [Pure]> {
  let summary = "Move one vector register to another";
  let description = [{
    Usually just moves one vector register to another, though some other modes of operation are also available.

    Example:
    ```mlir
    %result = sfpi.mov %src {mod1 = #sfpi<mov_mod1 COMPSIGN>} : vector<4x8xf32>
    ```
  }];
  let arguments = (ins SFPI_VectorType:$src, SFPI_MovMod1Attr:$mod1);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Arithmetic Operations
//===----------------------------------------------------------------------===//

def SFPI_AddOp : SFPI_BinaryOp<"add", [Pure, Commutative]> {
  let summary = "Vectorised floating-point addition";
  let description = [{
    Identical to SFPMAD, but is the preferred opcode when VA == 10, as this causes the computation to be lanewise FP32 VD = 1.0 * VB + VC (see the definition of LReg[10]).

    Example:
    ```mlir
    %result = sfpi.add %lhs, %rhs {mod1 = #sfpi<add_mod1 ARG_LREG_DST>} : vector<4x8xf32>
    ```
  }];
  let arguments = (ins SFPI_VectorType:$lhs, SFPI_VectorType:$rhs,
      SFPI_AddMod1Attr:$mod1);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) "
                       "`->` type($result)";
}

def SFPI_MulOp : SFPI_BinaryOp<"mul", [Pure, Commutative]> {
  let summary = "Vectorised floating-point multiply";
  let description = [{
    Identical to SFPMAD, but is the preferred opcode when VC == 9, as this causes the computation to be lanewise FP32 VD = VA * VB + 0 (see the definition of LReg[9]).

    Example:
    ```mlir
    %result = sfpi.mul %lhs, %rhs {mod1 = #sfpi<mad_mod1 OFFSET_NONE>} : vector<4x8xf32>
    ```
  }];
  let arguments = (ins SFPI_VectorType:$lhs, SFPI_VectorType:$rhs,
      SFPI_MadMod1Attr:$mod1);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) "
                       "`->` type($result)";
}

def SFPI_MadOp : SFPI_TernaryOp<"mad"> {
  let summary = "Vectorised floating-point multiply then add";
  let description = [{
    Performs lanewise FP32 VD = VA * VB + VC. In some modes, the VA and/or VD indices from the instruction bits are ignored, and instead come from the low four bits of LReg[7] (which allows these bits to potentially differ between lanes).

    Example:
    ```mlir
    %result = sfpi.mad %a, %b, %c {mod1 = #sfpi<mad_mod1 OFFSET_NONE>} : vector<4x8xf32>
    ```
  }];
  let arguments = (ins SFPI_VectorType:$a, SFPI_VectorType:$b,
      SFPI_VectorType:$c, SFPI_MadMod1Attr:$mod1);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$a `,` $b `,` $c attr-dict `:` type($a) `,` type($b) "
                       "`,` type($c) `->` type($result)";
}

def SFPI_DivP2Op : SFPI_UnaryOp<"divp2"> {
  let summary = "Adjust FP32 exponent bits";
  let description = [{
    Performs lanewise adjustment of FP32 exponent bits. The exponent bits are either set to the provided immediate, or the provided immediate is added to the exponent bits. In the latter case, no addition is performed if the starting value is infinity or NaN, and the result of the addition can wrap around. If software wishes to have exponent saturation rather than exponent wrapping, SFPMULI can be used instead.

    Example:
    ```mlir
    %result = sfpi.divp2 %src, %exp : vector<4x8xf32>
    ```
  }];
  let arguments = (ins SFPI_VectorType:$src, I32:$exp);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$src `,` $exp attr-dict `:` type($src) `,` type($exp) "
                       "`->` type($result)";
}

def SFPI_XiAddIOp : SFPI_BinaryOp<"xiadd_i"> {
  let summary = "Integer addition or subtraction with immediate";
  let description = [{
    Performs lanewise addition or subtraction between two vectors of 32-bit unsigned integers or 32-bit signed two's complement integers, or adds a signed 12-bit immediate to all lanes of such a vector. Lane flags can also be set based on the sign of the result.
  }];
  let arguments = (ins SFPI_VInt:$src, I32:$imm, SFPI_XiAddMod1Attr:$mod1);
  let results = (outs SFPI_VInt:$result);
  let assemblyFormat = "$src `,` $imm attr-dict `:` type($src) `,` type($imm) "
                       "`->` type($result)";
}

def SFPI_XiAddVOp : SFPI_BinaryOp<"xiadd_v"> {
  let summary = "Integer vector addition or subtraction";
  let description = [{
    Performs lanewise addition or subtraction between two vectors of 32-bit unsigned integers or 32-bit signed two's complement integers, or adds a signed 12-bit immediate to all lanes of such a vector. Lane flags can also be set based on the sign of the result.
  }];
  let arguments = (ins SFPI_VInt:$dst, SFPI_VInt:$src,
      SFPI_XiAddMod1Attr:$mod1);
  let results = (outs SFPI_VInt:$result);
  let assemblyFormat = "$dst `,` $src attr-dict `:` type($dst) `,` type($src) "
                       "`->` type($result)";
}

//===----------------------------------------------------------------------===//
// Bitwise Operations
//===----------------------------------------------------------------------===//

def SFPI_AndOp : SFPI_BinaryOp<"and", [Pure, Commutative]> {
  let summary = "Lanewise bitwise AND";
  let description = [{
    Performs lanewise bitwise-and between two vectors of 32-bit unsigned integers.
  }];
  let arguments = (ins SFPI_VectorType:$lhs, SFPI_VectorType:$rhs);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) "
                       "`->` type($result)";
}

def SFPI_OrOp : SFPI_BinaryOp<"or", [Pure, Commutative]> {
  let summary = "Lanewise bitwise OR";
  let description = [{
    Performs lanewise bitwise-or between two vectors of 32-bit unsigned integers.
  }];
  let arguments = (ins SFPI_VectorType:$lhs, SFPI_VectorType:$rhs);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) "
                       "`->` type($result)";
}

def SFPI_XorOp : SFPI_BinaryOp<"xor", [Pure, Commutative]> {
  let summary = "Lanewise bitwise exclusive-or";
  let description = [{
    Performs lanewise bitwise exclusive-or between two vectors of 32-bit unsigned integers.
  }];
  let arguments = (ins SFPI_VectorType:$lhs, SFPI_VectorType:$rhs);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) "
                       "`->` type($result)";
}

def SFPI_NotOp : SFPI_UnaryOp<"not"> {
  let summary = "Lanewise bitwise NOT";
  let description = [{
    Performs lanewise bitwise-not on a vector of 32-bit unsigned integers.
  }];
  let arguments = (ins SFPI_VectorType:$src);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Floating-Point Manipulation Operations
//===----------------------------------------------------------------------===//

def SFPI_SetExpIOp : SFPI_BinaryOp<"setexp_i"> {
  let summary = "Set exponent of SFPI vector with immediate";
  let description = [{
    Maps to sfpsetexp_i instruction. Sets the exponent bits using immediate value.
  }];
  let arguments = (ins I32:$imm, SFPI_VFloat:$src);
  let results = (outs SFPI_VFloat:$result);
  let assemblyFormat = "$imm `,` $src attr-dict `:` type($imm) `,` type($src) "
                       "`->` type($result)";
}

def SFPI_SetExpVOp : SFPI_BinaryOp<"setexp_v"> {
  let summary = "Set exponent of SFPI vector with immediate";
  let description = [{
    Maps to sfpsetexp_i instruction. Sets the exponent bits using immediate value.
  }];
  let arguments = (ins I32:$imm, SFPI_VFloat:$src);
  let results = (outs SFPI_VFloat:$result);
  let assemblyFormat = "$imm `,` $src attr-dict `:` type($imm) `,` type($src) "
                       "`->` type($result)";
}

def SFPI_SetManIOp : SFPI_BinaryOp<"setman_i"> {
  let summary = "Set mantissa of SFPI vector with immediate";
  let description = [{
    Maps to sfpsetman_i instruction. Sets the mantissa bits using immediate value.
  }];
  let arguments = (ins I32:$imm, SFPI_VFloat:$src);
  let results = (outs SFPI_VFloat:$result);
  let assemblyFormat = "$imm `,` $src attr-dict `:` type($imm) `,` type($src) "
                       "`->` type($result)";
}

def SFPI_SetManVOp : SFPI_BinaryOp<"setman_v"> {
  let summary = "Combine FP32 mantissa bits with sign and exponent";
  let description = [{
    Operating lanewise, combines the FP32 mantissa bits from one place (either an immediate, or part of the destination register) with the FP32 sign and exponent bits from somewhere else.
  }];
  let arguments = (ins SFPI_VFloat:$src, SFPI_VInt:$man);
  let results = (outs SFPI_VFloat:$result);
  let assemblyFormat = "$src `,` $man attr-dict `:` type($src) `,` type($man) "
                       "`->` type($result)";
}

def SFPI_SetSgnIOp : SFPI_BinaryOp<"setsgn_i"> {
  let summary = "Set sign of SFPI vector with immediate";
  let description = [{
    Maps to sfpsetsgn_i instruction. Sets the sign bits using immediate value.
  }];
  let arguments = (ins I32:$imm, SFPI_VFloat:$src);
  let results = (outs SFPI_VFloat:$result);
  let assemblyFormat = "$imm `,` $src attr-dict `:` type($imm) `,` type($src) "
                       "`->` type($result)";
}

def SFPI_SetSgnVOp : SFPI_BinaryOp<"setsgn_v"> {
  let summary = "Combine FP32 sign bit with exponent and mantissa";
  let description = [{
    Operating lanewise, combines the FP32 sign bit from one place (either an immediate, or part of the destination register) with the FP32 exponent and mantissa bits from somewhere else. Can instead be operating on sign-magnitude integers, combining the sign bit from one place (either an immediate, or part of the destination register) with the magnitude bits from somewhere else. In either case, when the sign bit comes from an immediate, this is computing either the absolute value or the negated absolute value.
  }];
  let arguments = (ins SFPI_VFloat:$src, SFPI_VInt:$sgn);
  let results = (outs SFPI_VFloat:$result);
  let assemblyFormat = "$src `,` $sgn attr-dict `:` type($src) `,` type($sgn) "
                       "`->` type($result)";
}

def SFPI_ExExpOp : SFPI_UnaryOp<"exexp"> {
  let summary = "Extract FP32 exponent bits";
  let description = [{
    Performs lanewise extraction of FP32 exponent bits, depositing them as an integer either in the range 0 through 255 or -127 through +128. When the latter range is used, the result is a two's complement integer. Lane flags can also be set based on the sign of the result.
  }];
  let arguments = (ins SFPI_VFloat:$src, SFPI_ExExpMod1Attr:$mod1);
  let results = (outs SFPI_VInt:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def SFPI_ExManOp : SFPI_UnaryOp<"exman"> {
  let summary = "Extract FP32 mantissa bits";
  let description = [{
    Performs lanewise extraction of FP32 mantissa bits, depositing them in the low 23 bits. The next significant bit can be set to either 0 or 1, and then the top 8 bits are always set to zero.
  }];
  let arguments = (ins SFPI_VFloat:$src, SFPI_ExManMod1Attr:$mod1);
  let results = (outs SFPI_VInt:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def SFPI_AbsOp : SFPI_UnaryOp<"abs"> {
  let summary = "Vectorised floating-point or two's complement absolute value";
  let description = [{
    Performs lanewise absolute value on a vector of two's complement integers or a vector of FP32. In the latter case, -NaN is left as -NaN rather than becoming +NaN. To instead take the absolute value on a vector of sign-magnitude integers, or a vector of FP32 without special case handling of NaN, SFPSETSGN can be used.
  }];
  let arguments = (ins SFPI_VFloat:$src, SFPI_AbsMod1Attr:$mod1);
  let results = (outs SFPI_VFloat:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def SFPI_LzOp : SFPI_UnaryOp<"lz"> {
  let summary = "Count leading zeroes on vector";
  let description = [{
    Performs lanewise count leading zeroes (lzcnt) on a vector of 32-bit unsigned integers or on the magnitudes of a vector of sign-magnitude integers. Inputs equal to zero are considered to have 32 leading zeroes. Lane flags can also be set based on whether the input was equal to zero.
  }];
  let arguments = (ins SFPI_VFloat:$src, SFPI_LzMod1Attr:$mod1);
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
  let arguments = (ins SFPI_VFloat:$lhs, SFPI_VFloat:$rhs,
      SFPI_XSCmpMod1Attr:$mod1);
  let results = (outs I1:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) "
                       "`->` type($result)";
}

def SFPI_XfCmpVOp : SFPI_ComparisonOp<"xfcmpv"> {
  let summary = "SFPI vector floating-point comparison";
  let description = [{
    Maps to sfpxfcmpv instruction. Performs vector floating-point comparison.
  }];
  let arguments = (ins SFPI_VFloat:$lhs, SFPI_VFloat:$rhs,
      SFPI_XCmpMod1Attr:$mod1);
  let results = (outs SFPI_VInt:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) "
                       "`->` type($result)";
}

def SFPI_XiCmpSOp : SFPI_ComparisonOp<"xicmps"> {
  let summary = "SFPI scalar integer comparison";
  let description = [{
    Maps to sfpxicmps instruction. Performs scalar integer comparison.
  }];
  let arguments = (ins SFPI_VInt:$lhs, I32:$rhs, SFPI_XSCmpMod1Attr:$mod1);
  let results = (outs I1:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) "
                       "`->` type($result)";
}

def SFPI_XiCmpVOp : SFPI_ComparisonOp<"xicmpv"> {
  let summary = "SFPI vector integer comparison";
  let description = [{
    Maps to sfpxicmpv instruction. Performs vector integer comparison.
  }];
  let arguments = (ins SFPI_VInt:$lhs, SFPI_VInt:$rhs, SFPI_XCmpMod1Attr:$mod1);
  let results = (outs SFPI_VInt:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) "
                       "`->` type($result)";
}

//===----------------------------------------------------------------------===//
// Type Conversion Operations
//===----------------------------------------------------------------------===//

def SFPI_CastOp : SFPI_ConversionOp<"cast"> {
  let summary = "Convert sign-magnitude integers to FP32";
  let description = [{
    Performs lanewise conversion of 32-bit sign-magnitude integers to FP32. The conversion is exact for inputs |x| ≤ 2²⁴. Inputs outside this range require rounding, which can either be stochastic or round to nearest with ties to even.
  }];
  let arguments = (ins SFPI_VectorType:$src, SFPI_CastMod1Attr:$mod1);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Specialized Operations
//===----------------------------------------------------------------------===//

def SFPI_ShftIOp : SFPI_Op<"shft_i"> {
  let summary = "Bitwise shift with immediate amount";
  let description = [{
    Performs lanewise bitwise (logical) shift left or right on a vector of 32-bit unsigned integers with immediate shift amount. There is no instruction for performing arithmetic right shift; software needs to construct this from a sequence of instructions if it requires it.
  }];
  let arguments = (ins SFPI_VectorType:$dst, I32:$imm);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$dst `,` $imm attr-dict `:` type($dst) `,` type($imm) "
                       "`->` type($result)";
}

def SFPI_ShftVOp : SFPI_Op<"shft_v"> {
  let summary = "Bitwise shift with vector amount";
  let description = [{
    Performs lanewise bitwise (logical) shift left or right on a vector of 32-bit unsigned integers with vector shift amounts. There is no instruction for performing arithmetic right shift; software needs to construct this from a sequence of instructions if it requires it.
  }];
  let arguments = (ins SFPI_VectorType:$dst, SFPI_VectorType:$src);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$dst `,` $src attr-dict `:` type($dst) `,` type($src) "
                       "`->` type($result)";
}

def SFPI_Shft2IOp : SFPI_Op<"shft2_i"> {
  let summary = "Bitwise shift or lane shuffle with immediate";
  let description = [{
    Performs some kind of bitwise shift within vector lanes, or some kind of shuffle of vector lanes. The exact behaviour is determined by the Mod1 field.
  }];
  let arguments = (ins SFPI_VectorType:$dst, I32:$imm);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$dst `,` $imm attr-dict `:` type($dst) `,` type($imm) "
                       "`->` type($result)";
}

def SFPI_Shft2VOp : SFPI_Op<"shft2_v"> {
  let summary = "Bitwise shift or lane shuffle with vector";
  let description = [{
    Performs some kind of bitwise shift within vector lanes, or some kind of shuffle of vector lanes. The exact behaviour is determined by the Mod1 field.
  }];
  let arguments = (ins SFPI_VectorType:$dst, SFPI_VectorType:$src);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$dst `,` $src attr-dict `:` type($dst) `,` type($src) "
                       "`->` type($result)";
}

def SFPI_Shft2GOp : SFPI_Op<"shft2_g"> {
  let summary = "Bitwise shift or lane shuffle gather";
  let description = [{
    Performs some kind of bitwise shift within vector lanes, or some kind of shuffle of vector lanes. The exact behaviour is determined by the Mod1 field.
  }];
  let arguments = (ins SFPI_VectorType:$l0, SFPI_VectorType:$l1,
      SFPI_VectorType:$l2, SFPI_VectorType:$l3, SFPI_Shft2Mod1Attr:$mod1);
  let results = (outs);
  let assemblyFormat = "$l0 `,` $l1 `,` $l2 `,` $l3 attr-dict `:` type($l0) "
                       "`,` type($l1) `,` type($l2) `,` type($l3)";
}

def SFPI_Shft2GEOp : SFPI_Op<"shft2_ge"> {
  let summary = "SFPI shift2 gather-extract operation";
  let description = [{
    Maps to sfpshft2_ge instruction. Performs shift2 gather-extract operation.
  }];
  let arguments = (ins SFPI_VectorType:$src, SFPI_VectorType:$l0,
      SFPI_VectorType:$l1, SFPI_VectorType:$l2, SFPI_VectorType:$l3);
  let results = (outs);
  let assemblyFormat =
      "$src `,` $l0 `,` $l1 `,` $l2 `,` $l3 attr-dict `:` type($src) `,` "
      "type($l0) `,` type($l1) `,` type($l2) `,` type($l3)";
}

def SFPI_Shft2EOp : SFPI_Op<"shft2_e"> {
  let summary = "Bitwise shift or lane shuffle extract";
  let description = [{
    Performs some kind of bitwise shift within vector lanes, or some kind of shuffle of vector lanes. The exact behaviour is determined by the Mod1 field.
  }];
  let arguments = (ins SFPI_VectorType:$src, SFPI_Shft2Mod1Attr:$mod1);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def SFPI_StochRndIOp : SFPI_Op<"stochrnd_i"> {
  let summary = "SFPI stochastic rounding with immediate";
  let description = [{
    Maps to sfpstochrnd_i instruction. Performs stochastic rounding with immediate value.
  }];
  let arguments = (ins SFPI_StochRndRndAttr:$mode, I32:$imm8, SFPI_VFloat:$srcc,
      SFPI_StochRndMod1Attr:$mod1);
  let results = (outs SFPI_VFloat:$result);
  let assemblyFormat = "$imm8 `,` $srcc attr-dict `:` type($imm8) `,` "
                       "type($srcc) `->` type($result)";
}

def SFPI_StochRndVOp : SFPI_UnaryOp<"stochrnd_v"> {
  let summary = "Stochastic rounding with three distinct modes";
  let description = [{
    This instruction has three distinct modes (references three separate documentation files for Float-Float, Float-Int, and Int-Int conversions).
  }];
  let arguments = (ins SFPI_StochRndRndAttr:$mode, I32:$imm8, SFPI_VFloat:$srcc,
      SFPI_StochRndMod1Attr:$mod1);
  let results = (outs SFPI_VFloat:$result);
  let assemblyFormat = "$imm8 `,` $srcc attr-dict `:` type($imm8) `,` "
                       "type($srcc) `->` type($result)";
}

def SFPI_LutOp : SFPI_UnaryOp<"lut"> {
  let summary = "FP32 multiply then add lookup table";
  let description = [{
    Operating lanewise, performs one of the FP32 multiply then add variants from the following table.
  }];
  let arguments = (ins SFPI_VectorType:$l0, SFPI_VectorType:$l1,
      SFPI_VectorType:$l2, SFPI_VectorType:$l3, SFPI_LutMod0Attr:$mod0);
  let results = (outs SFPI_VFloat:$result);
  let assemblyFormat =
      "$l0 `,` $l1 `,` $l2 `,` $l3 attr-dict `:` type($l0) `,` type($l1) `,` "
      "type($l2) `,` type($l3) `->` type($result)";
}

def SFPI_LutFp32_3ROp : SFPI_Op<"lutfp32_3r"> {
  let summary = "SFPI FP32 lookup table 3-register";
  let description = [{
    Maps to sfplutfp32_3r instruction. Performs FP32 lookup table with 3 registers.
  }];
  let arguments = (ins SFPI_VectorType:$l0, SFPI_VectorType:$l1,
      SFPI_VectorType:$l2, SFPI_VectorType:$l3, SFPI_LutFp32Mod0Attr:$mod0);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat =
      "$l0 `,` $l1 `,` $l2 `,` $l3 attr-dict `:` type($l0) `,` type($l1) `,` "
      "type($l2) `,` type($l3) `->` type($result)";
}

def SFPI_LutFp32_6ROp : SFPI_Op<"lutfp32_6r"> {
  let summary = "SFPI FP32 lookup table 6-register";
  let description = [{
    Maps to sfplutfp32_6r instruction. Performs FP32 lookup table with 6 registers.
  }];
  let arguments = (ins SFPI_VectorType:$l0, SFPI_VectorType:$l1,
      SFPI_VectorType:$l2, SFPI_VectorType:$l4, SFPI_VectorType:$l5,
      SFPI_VectorType:$l6, SFPI_VectorType:$l3, SFPI_LutFp32Mod0Attr:$mod0);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat =
      "$l0 `,` $l1 `,` $l2 `,` $l4 `,` $l5 `,` $l6 `,` $l3 attr-dict `:` "
      "type($l0) `,` type($l1) `,` type($l2) `,` type($l4) `,` type($l5) `,` "
      "type($l6) `,` type($l3) `->` type($result)";
}

def SFPI_SwapOp : SFPI_BinaryOp<"swap"> {
  let summary = "Vector swap, min/max, or conditional swap";
  let description = [{
    In its simplest mode, SFPSWAP swaps the contents of two vectors. In the next mode up, it takes two vectors of FP32 or two vectors of 32-bit sign-magnitude integers, simultaneously computes the lanewise minimum and lanewise maximum, and returns both vectors. In its most complex mode, it can be used for argmin+argmax: conditionally swap a pair of vectors (lanewise) based on the results of comparing a different pair of vectors (lanewise).

    Example:
    ```mlir
    %result = sfpi.swap %dst, %src {mode = #sfpi<swap_mode SWAP>} : vector<4x8xf32>
    ```
  }];
  let arguments = (ins SFPI_VectorType:$dst, SFPI_VectorType:$src,
      SFPI_SwapMod1Attr:$mod1);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$dst `,` $src attr-dict `:` type($dst) `,` type($src) "
                       "`->` type($result)";
}

def SFPI_TranspOp : SFPI_Op<"transp"> {
  let summary = "Matrix transpose within columns";
  let description = [{
    Viewing each vector register as a 4x8 grid of lanes, a group of four registers is stacked vertically to form a 16x8 grid, then a transpose is performed within each column: the 16 values in the column are reshaped to 4x4, transposed, then reshaped back. This is applied to LReg[0:4] and also to LReg[4:8].

    Example:
    ```mlir
    sfpi.transp %l0, %l1, %l2, %l3 : vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32>
    ```
  }];
  let arguments = (ins SFPI_VectorType:$l0, SFPI_VectorType:$l1,
      SFPI_VectorType:$l2, SFPI_VectorType:$l3);
  let results = (outs);
  let assemblyFormat = "$l0 `,` $l1 `,` $l2 `,` $l3 attr-dict `:` type($l0) "
                       "`,` type($l1) `,` type($l2) `,` type($l3)";
}

//===----------------------------------------------------------------------===//
// Condition Code Operations
//===----------------------------------------------------------------------===//

def SFPI_SetCCIOp : SFPI_Op<"setcc_i"> {
  let summary = "SFPI set condition code with immediate";
  let description = [{
    Maps to sfpsetcc_i instruction. Sets condition code using immediate value.
  }];
  let arguments = (ins I32:$imm, SFPI_SetCCMod1Attr:$mod1);
  let results = (outs);
  let assemblyFormat = "$imm attr-dict `:` type($imm)";
}

def SFPI_SetCCVOp : SFPI_Op<"setcc_v"> {
  let summary = "Lanewise comparison and set lane flags";
  let description = [{
    Operating lanewise, performs a comparison VC < 0 or VC != 0 or VC >= 0 or VC == 0, and writes the result of the comparison to LaneFlags. The elements of the input vector can be any type of 32-bit data, though software should ensure that sign-magnitude integers and FP32 have had negative zero flushed to positive zero.
  }];
  let arguments = (ins SFPI_VectorType:$src, SFPI_SetCCMod1Attr:$mod1);
  let results = (outs);
  let assemblyFormat = "$src attr-dict `:` type($src)";
}

def SFPI_EnCCOp : SFPI_Op<"encc"> {
  let summary = "Set lane predication mask usage";
  let description = [{
    Used to set UseLaneFlagsForLaneEnable. When UseLaneFlagsForLaneEnable is true, the per-lane LaneFlags boolean acts as a predication mask on Vector Unit (SFPU) instructions, allowing some lanes to be active whilst others are not.
  }];
  let arguments = (ins I32:$imm, SFPI_EnCCMod1Attr:$mod1);
  let results = (outs);
  let assemblyFormat = "$imm attr-dict `:` type($imm)";
}

def SFPI_PushCOp : SFPI_Op<"pushc"> {
  let summary = "Push lane predication masks to stack";
  let description = [{
    Pushes a copy of the current lane predication masks (per-lane LaneFlags and UseLaneFlagsForLaneEnable) onto the vector conditional execution stack. The stack has a maximum capacity of eight elements, and this instruction must not be used if the stack is already full.
  }];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def SFPI_PopCOp : SFPI_Op<"popc"> {
  let summary = "Pop or peek from vector conditional execution stack";
  let description = [{
    In its simplest mode, SFPPOPC pops from the vector conditional execution stack, replacing LaneFlags and UseLaneFlagsForLaneEnable with values popped from the stack. In its more complex modes, SFPPOPC peeks from the vector conditional execution stack, mutating LaneFlags and UseLaneFlagsForLaneEnable in various ways based on the contents of the top of the stack. The simple mode must not be used when the stack is empty. Due to a hardware bug, the complex modes should not be used when the stack is full.
  }];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def SFPI_CompCOp : SFPI_Op<"compc"> {
  let summary = "Implement else for SIMT if/else";
  let description = [{
    Used to implement else when a compiler maps SIMT if / else onto the vector conditional execution stack.
  }];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Boolean and Conditional Operations
//===----------------------------------------------------------------------===//

def SFPI_XBoolOp : SFPI_Op<"xbool"> {
  let summary = "SFPI extended boolean operation";
  let description = [{
    Maps to sfpxbool instruction. Performs boolean operations on vectors.
  }];
  let arguments = (ins I32:$t, SFPI_VectorType:$a, SFPI_VectorType:$b);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat = "$t `,` $a `,` $b attr-dict `:` type($t) `,` type($a) "
                       "`,` type($b) `->` type($result)";
}

def SFPI_XCondBOp : SFPI_Op<"xcondb"> {
  let summary = "SFPI extended conditional branch";
  let description = [{
    Maps to sfpxcondb instruction. Performs conditional branch operations.
  }];
  let arguments = (ins SFPI_VectorType:$s, I32:$i);
  let results = (outs);
  let assemblyFormat = "$s `,` $i attr-dict `:` type($s) `,` type($i)";
}

def SFPI_XCondIOp : SFPI_Op<"xcondi"> {
  let summary = "SFPI extended conditional immediate";
  let description = [{
    Maps to sfpxcondi instruction. Performs conditional operations with immediate.
  }];
  let arguments = (ins I32:$i);
  let results = (outs);
  let assemblyFormat = "$i attr-dict `:` type($i)";
}

def SFPI_XVifOp : SFPI_Op<"xvif"> {
  let summary = "SFPI extended vector if";
  let description = [{
    Maps to sfpxvif instruction. Performs vector conditional operations.
  }];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Register Management Operations
//===----------------------------------------------------------------------===//

def SFPI_AssignLRegOp : SFPI_Op<"assignlreg"> {
  let summary = "SFPI assign L register";
  let description = [{
    Maps to sfpassignlreg instruction. Assigns value to L register.
  }];
  let arguments = (ins I32:$lr);
  let results = (outs);
  let assemblyFormat = "$lr attr-dict `:` type($lr)";
}

def SFPI_AssignLvOp : SFPI_Op<"assign_lv"> {
  let summary = "SFPI assign live vector";
  let description = [{
    Maps to sfpassign_lv instruction. Assigns live vector value.
  }];
  let arguments = (ins SFPI_VectorType:$v, I32:$in);
  let results = (outs SFPI_VectorType:$result);
  let assemblyFormat =
      "$v `,` $in attr-dict `:` type($v) `,` type($in) `->` type($result)";
}

def SFPI_PreserveLRegOp : SFPI_Op<"preservelreg"> {
  let summary = "SFPI preserve L register";
  let description = [{
    Maps to sfppreservelreg instruction. Preserves L register value.
  }];
  let arguments = (ins I32:$x, I32:$n);
  let results = (outs);
  let assemblyFormat = "$x `,` $n attr-dict `:` type($x) `,` type($n)";
}

//===----------------------------------------------------------------------===//
// Configuration Operations
//===----------------------------------------------------------------------===//

def SFPI_ConfigVOp : SFPI_Op<"config_v"> {
  let summary = "Configure LReg, LaneConfig, or LoadMacroConfig";
  let description = [{
    In its simplest mode, SFPCONFIG moves values to LReg[11] through LReg[14]. In the next mode up, SFPCONFIG manipulates LaneConfig. In its most complex modes, it manipulates LoadMacroConfig, which is used by the SFPLOADMACRO instruction.

    Example:
    ```mlir
    sfpi.config %value {dest = #sfpi<config_dest SFPU_CTRL>} : i32
    ```
  }];
  let arguments = (ins I32:$value, SFPI_ConfigDestAttr:$dest);
  let results = (outs);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

#endif // TTMLIR_DIALECT_SFPI_IR_SFPIOPS_TD
