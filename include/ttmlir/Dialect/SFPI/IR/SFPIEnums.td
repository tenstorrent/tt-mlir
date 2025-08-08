// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_SFPI_IR_SFPIENUMS_TD
#define TTMLIR_DIALECT_SFPI_IR_SFPIENUMS_TD

include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// SFPI Dialect Enums
// Source:
// https://github.com/tenstorrent/sfpi/blob/main/include/wormhole/sfpi_hw.h
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SFPI Load Format Enums
//===----------------------------------------------------------------------===//

def SFPI_LOAD_MOD0_FMT_SRCB : I32EnumAttrCase<"FMT_SRCB", 0, "FMT_SRCB">;
def SFPI_LOAD_MOD0_FMT_FP16A : I32EnumAttrCase<"FMT_FP16A", 1, "FMT_FP16A">;
def SFPI_LOAD_MOD0_FMT_FP16B : I32EnumAttrCase<"FMT_FP16B", 2, "FMT_FP16B">;
def SFPI_LOAD_MOD0_FMT_FP32 : I32EnumAttrCase<"FMT_FP32", 3, "FMT_FP32">;
def SFPI_LOAD_MOD0_FMT_INT32_TO_SM
    : I32EnumAttrCase<"FMT_INT32_TO_SM", 12, "FMT_INT32_TO_SM">;

def SFPI_LOAD_MOD0
    : I32EnumAttr<"LOAD_MOD0", "SFPI Load Format",
                  [SFPI_LOAD_MOD0_FMT_SRCB, SFPI_LOAD_MOD0_FMT_FP16A,
                   SFPI_LOAD_MOD0_FMT_FP16B, SFPI_LOAD_MOD0_FMT_FP32,
                   SFPI_LOAD_MOD0_FMT_INT32_TO_SM]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
}

//===----------------------------------------------------------------------===//
// SFPI Load Address Mode Enums
//===----------------------------------------------------------------------===//

def SFPI_LOAD_ADDR_MODE_NOINC : I32EnumAttrCase<"NOINC", 3, "NOINC">;

def SFPI_LOAD_ADDR_MODE
    : I32EnumAttr<"LOAD_ADDR_MODE",
                  "SFPI Load Address Mode", [SFPI_LOAD_ADDR_MODE_NOINC]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
}

//===----------------------------------------------------------------------===//
// SFPI Store Format Enums
//===----------------------------------------------------------------------===//

def SFPI_STORE_MOD0_FMT_SRCB : I32EnumAttrCase<"FMT_SRCB", 0, "FMT_SRCB">;
def SFPI_STORE_MOD0_FMT_FP16A : I32EnumAttrCase<"FMT_FP16A", 1, "FMT_FP16A">;
def SFPI_STORE_MOD0_FMT_FP16B : I32EnumAttrCase<"FMT_FP16B", 2, "FMT_FP16B">;
def SFPI_STORE_MOD0_FMT_FP32 : I32EnumAttrCase<"FMT_FP32", 3, "FMT_FP32">;
def SFPI_STORE_MOD0_FMT_INT32_TO_SM
    : I32EnumAttrCase<"FMT_INT32_TO_SM", 12, "FMT_INT32_TO_SM">;

def SFPI_STORE_MOD0
    : I32EnumAttr<"STORE_MOD0", "SFPI Store Format",
                  [SFPI_STORE_MOD0_FMT_SRCB, SFPI_STORE_MOD0_FMT_FP16A,
                   SFPI_STORE_MOD0_FMT_FP16B, SFPI_STORE_MOD0_FMT_FP32,
                   SFPI_STORE_MOD0_FMT_INT32_TO_SM]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
}

//===----------------------------------------------------------------------===//
// SFPI Store Address Mode Enums
//===----------------------------------------------------------------------===//

def SFPI_STORE_ADDR_MODE_NOINC : I32EnumAttrCase<"NOINC", 3, "NOINC">;

def SFPI_STORE_ADDR_MODE
    : I32EnumAttr<"STORE_ADDR_MODE",
                  "SFPI Store Address Mode", [SFPI_STORE_ADDR_MODE_NOINC]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
}

//===----------------------------------------------------------------------===//
// SFPI Move Mod1 Enums
//===----------------------------------------------------------------------===//

def SFPI_MOV_MOD1_COMPSIGN : I32EnumAttrCase<"COMPSIGN", 1, "COMPSIGN">;

def SFPI_MOV_MOD1
    : I32EnumAttr<"MOV_MOD1", "SFPI Move Mod1 Mode", [SFPI_MOV_MOD1_COMPSIGN]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
}

//===----------------------------------------------------------------------===//
// SFPI Mad Mod1 Enums
//===----------------------------------------------------------------------===//

def SFPI_MAD_MOD1_OFFSET_NONE
    : I32EnumAttrCase<"OFFSET_NONE", 0, "OFFSET_NONE">;

def SFPI_MAD_MOD1
    : I32EnumAttr<"MAD_MOD1",
                  "SFPI Mad Mod1 Mode", [SFPI_MAD_MOD1_OFFSET_NONE]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
}

//===----------------------------------------------------------------------===//
// SFPI Load Immediate Mode Enums
//===----------------------------------------------------------------------===//

def SFPI_LOADI_MOD0_FLOATB : I32EnumAttrCase<"FLOATB", 0, "FLOATB">;
def SFPI_LOADI_MOD0_FLOATA : I32EnumAttrCase<"FLOATA", 1, "FLOATA">;
def SFPI_LOADI_MOD0_USHORT : I32EnumAttrCase<"USHORT", 2, "USHORT">;
def SFPI_LOADI_MOD0_SHORT : I32EnumAttrCase<"SHORT", 4, "SHORT">;
def SFPI_LOADI_MOD0_UPPER : I32EnumAttrCase<"UPPER", 8, "UPPER">;
def SFPI_LOADI_MOD0_LOWER : I32EnumAttrCase<"LOWER", 10, "LOWER">;

def SFPI_LOADI_MOD0
    : I32EnumAttr<"LOADI_MOD0", "SFPI Load Immediate Mod0",
                  [SFPI_LOADI_MOD0_FLOATB, SFPI_LOADI_MOD0_FLOATA,
                   SFPI_LOADI_MOD0_USHORT, SFPI_LOADI_MOD0_SHORT,
                   SFPI_LOADI_MOD0_UPPER, SFPI_LOADI_MOD0_LOWER]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
}

//===----------------------------------------------------------------------===//
// SFPI Extended Load Immediate Mode Enums
//===----------------------------------------------------------------------===//

def SFPI_XLOADI_MOD0_INT32 : I32EnumAttrCase<"INT32", 16, "INT32">;
def SFPI_XLOADI_MOD0_UINT32 : I32EnumAttrCase<"UINT32", 17, "UINT32">;
def SFPI_XLOADI_MOD0_FLOAT : I32EnumAttrCase<"FLOAT", 18, "FLOAT">;

def SFPI_XLOADI_MOD0
    : I32EnumAttr<"XLOADI_MOD0", "SFPI Extended Load Immediate Mod0",
                  [SFPI_XLOADI_MOD0_INT32, SFPI_XLOADI_MOD0_UINT32,
                   SFPI_XLOADI_MOD0_FLOAT]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
}

//===----------------------------------------------------------------------===//
// SFPI Extended Mantissa Mode Enums
//===----------------------------------------------------------------------===//

def SFPI_EXMAN_MOD1_PAD8 : I32EnumAttrCase<"PAD8", 0, "PAD8">;
def SFPI_EXMAN_MOD1_PAD9 : I32EnumAttrCase<"PAD9", 1, "PAD9">;

def SFPI_EXMAN_MOD1
    : I32EnumAttr<"EXMAN_MOD1", "SFPI Extended Mantissa Mode",
                  [SFPI_EXMAN_MOD1_PAD8, SFPI_EXMAN_MOD1_PAD9]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
}

//===----------------------------------------------------------------------===//
// SFPI Extended Exponent Mode Enums
//===----------------------------------------------------------------------===//

def SFPI_EXEXP_MOD1_DEBIAS : I32BitEnumAttrCase<"DEBIAS", 0, "DEBIAS">;
def SFPI_EXEXP_MOD1_NODEBIAS : I32BitEnumAttrCase<"NODEBIAS", 1, "NODEBIAS">;
def SFPI_EXEXP_MOD1_SET_CC_SGN_EXP
    : I32BitEnumAttrCase<"SET_CC_SGN_EXP", 2, "SET_CC_SGN_EXP">;
def SFPI_EXEXP_MOD1_SET_CC_COMP_EXP
    : I32BitEnumAttrCase<"SET_CC_COMP_EXP", 8, "SET_CC_COMP_EXP">;
def SFPI_EXEXP_MOD1_SET_CC_SGN_COMP_EXP
    : I32BitEnumAttrCase<"SET_CC_SGN_COMP_EXP", 10, "SET_CC_SGN_COMP_EXP">;

def SFPI_EXEXP_MOD1
    : I32BitEnumAttr<"EXEXP_MOD1", "SFPI Extended Exponent Mode",
                     [SFPI_EXEXP_MOD1_DEBIAS, SFPI_EXEXP_MOD1_NODEBIAS,
                      SFPI_EXEXP_MOD1_SET_CC_SGN_EXP,
                      SFPI_EXEXP_MOD1_SET_CC_COMP_EXP,
                      SFPI_EXEXP_MOD1_SET_CC_SGN_COMP_EXP]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
  let separator = "|";
}

//===----------------------------------------------------------------------===//
// SFPI Absolute Value Mode Enums
//===----------------------------------------------------------------------===//

def SFPI_ABS_MOD1_INT : I32EnumAttrCase<"INT", 0, "INT">;
def SFPI_ABS_MOD1_FLOAT : I32EnumAttrCase<"FLOAT", 1, "FLOAT">;

def SFPI_ABS_MOD1 : I32EnumAttr<"ABS_MOD1", "SFPI Absolute Value Mod1",
                                [SFPI_ABS_MOD1_INT, SFPI_ABS_MOD1_FLOAT]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
}

//===----------------------------------------------------------------------===//
// SFPI Add Mod1 Enums (SFPIADD_MOD1_*)
//===----------------------------------------------------------------------===//

def SFPI_ADD_MOD1_ARG_LREG_DST
    : I32BitEnumAttrCase<"ARG_LREG_DST", 0, "ARG_LREG_DST">;
def SFPI_ADD_MOD1_ARG_IMM : I32BitEnumAttrCase<"ARG_IMM", 1, "ARG_IMM">;
def SFPI_ADD_MOD1_ARG_2SCOMP_LREG_DST
    : I32BitEnumAttrCase<"ARG_2SCOMP_LREG_DST", 2, "ARG_2SCOMP_LREG_DST">;
// def SFPI_ADD_MOD1_CC_LT0 : I32BitEnumAttrCase<"CC_LT0", 0, "CC_LT0">;
def SFPI_ADD_MOD1_CC_NONE : I32BitEnumAttrCase<"CC_NONE", 4, "CC_NONE">;
def SFPI_ADD_MOD1_CC_GTE0 : I32BitEnumAttrCase<"CC_GTE0", 8, "CC_GTE0">;

def SFPI_ADD_MOD1
    : I32BitEnumAttr<"ADD_MOD1", "SFPI Add Mod1 Mode",
                     [SFPI_ADD_MOD1_ARG_LREG_DST, SFPI_ADD_MOD1_ARG_IMM,
                      SFPI_ADD_MOD1_ARG_2SCOMP_LREG_DST,
                      // SFPI_ADD_MOD1_CC_LT0,
                      SFPI_ADD_MOD1_CC_NONE, SFPI_ADD_MOD1_CC_GTE0]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
  let separator = "|";
}

//===----------------------------------------------------------------------===//
// SFPI Integer Add Mode Enums
//===----------------------------------------------------------------------===//

def SFPI_XIADD_MOD1_SIGNED : I32BitEnumAttrCase<"SIGNED", 8, "SIGNED">;
def SFPI_XIADD_MOD1_IS_SUB : I32BitEnumAttrCase<"IS_SUB", 16, "IS_SUB">;
def SFPI_XIADD_MOD1_12BIT : I32BitEnumAttrCase<"BIT12", 32, "BIT12">;
def SFPI_XIADD_MOD1_16BIT : I32BitEnumAttrCase<"BIT16", 64, "BIT16">;

def SFPI_XIADD_MOD1
    : I32BitEnumAttr<"XIADD_MOD1", "SFPI Integer Add Mode",
                     [SFPI_XIADD_MOD1_SIGNED, SFPI_XIADD_MOD1_IS_SUB,
                      SFPI_XIADD_MOD1_12BIT, SFPI_XIADD_MOD1_16BIT]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
  let separator = "|";
}

//===----------------------------------------------------------------------===//
// SFPI Comparison Condition Enums
//===----------------------------------------------------------------------===//

def SFPI_XCMP_MOD1_CC_NONE : I32EnumAttrCase<"CC_NONE", 0, "CC_NONE">;
def SFPI_XCMP_MOD1_CC_LT : I32EnumAttrCase<"CC_LT", 1, "CC_LT">;
def SFPI_XCMP_MOD1_CC_EQ : I32EnumAttrCase<"CC_EQ", 2, "CC_EQ">;
def SFPI_XCMP_MOD1_CC_GTE : I32EnumAttrCase<"CC_GTE", 3, "CC_GTE">;
def SFPI_XCMP_MOD1_CC_NE : I32EnumAttrCase<"CC_NE", 4, "CC_NE">;
def SFPI_XCMP_MOD1_CC_LTE : I32EnumAttrCase<"CC_LTE", 5, "CC_LTE">;
def SFPI_XCMP_MOD1_CC_GT : I32EnumAttrCase<"CC_GT", 6, "CC_GT">;

def SFPI_XCMP_MOD1 : I32EnumAttr<"XCMP_MOD1", "SFPI Compare Condition",
                                 [SFPI_XCMP_MOD1_CC_NONE, SFPI_XCMP_MOD1_CC_LT,
                                  SFPI_XCMP_MOD1_CC_EQ, SFPI_XCMP_MOD1_CC_GTE,
                                  SFPI_XCMP_MOD1_CC_NE, SFPI_XCMP_MOD1_CC_LTE,
                                  SFPI_XCMP_MOD1_CC_GT]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
}

def SFPI_XSCMP_MOD1_FMT_A : I32EnumAttrCase<"FMT_A", 8, "FMT_A">;
def SFPI_XSCMP_MOD1_FMT_B : I32EnumAttrCase<"FMT_B", 16, "FMT_B">;
def SFPI_XSCMP_MOD1_FMT_FLOAT : I32EnumAttrCase<"FMT_FLOAT", 32, "FMT_FLOAT">;

def SFPI_XSCMP_MOD1 : I32EnumAttr<"XSCMP_MOD1", "SFPI Compare Format",
                                  [SFPI_XSCMP_MOD1_FMT_A, SFPI_XSCMP_MOD1_FMT_B,
                                   SFPI_XSCMP_MOD1_FMT_FLOAT]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
}

def SFPI_XBOOL_MOD1_OR : I32EnumAttrCase<"OR", 1, "OR">;
def SFPI_XBOOL_MOD1_AND : I32EnumAttrCase<"AND", 2, "AND">;
def SFPI_XBOOL_MOD1_NOT : I32EnumAttrCase<"NOT", 3, "NOT">;

def SFPI_XBOOL_MOD1 : I32EnumAttr<"XBOOL_MOD1", "SFPI Bool Mode",
                                  [SFPI_XBOOL_MOD1_OR, SFPI_XBOOL_MOD1_AND,
                                   SFPI_XBOOL_MOD1_NOT]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
}

def SFPI_SETCC_MOD1_LREG_LT0 : I32BitEnumAttrCase<"LREG_LT0", 0, "LREG_LT0">;
def SFPI_SETCC_MOD1_IMM_BIT0 : I32BitEnumAttrCase<"IMM_BIT0", 1, "IMM_BIT0">;
def SFPI_SETCC_MOD1_LREG_NE0 : I32BitEnumAttrCase<"LREG_NE0", 2, "LREG_NE0">;
def SFPI_SETCC_MOD1_LREG_GTE0 : I32BitEnumAttrCase<"LREG_GTE0", 4, "LREG_GTE0">;
def SFPI_SETCC_MOD1_LREG_EQ0 : I32BitEnumAttrCase<"LREG_EQ0", 6, "LREG_EQ0">;
def SFPI_SETCC_MOD1_COMP : I32BitEnumAttrCase<"COMP", 8, "COMP">;

def SFPI_SETCC_MOD1
    : I32BitEnumAttr<"SETCC_MOD1", "SFPI Bool Mode",
                     [SFPI_SETCC_MOD1_LREG_LT0, SFPI_SETCC_MOD1_IMM_BIT0,
                      SFPI_SETCC_MOD1_LREG_NE0, SFPI_SETCC_MOD1_LREG_GTE0,
                      SFPI_SETCC_MOD1_LREG_EQ0, SFPI_SETCC_MOD1_COMP]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
  let separator = "|";
}

def SFPI_ENCC_IMM12_NEITHER : I32EnumAttrCase<"NEITHER", 0, "NEITHER">;
def SFPI_ENCC_IMM12_BOTH : I32EnumAttrCase<"BOTH", 3, "BOTH">;

def SFPI_ENCC_IMM12
    : I32EnumAttr<
          "ENCC_IMM12",
          "SFPI Bool Mode", [SFPI_ENCC_IMM12_NEITHER, SFPI_ENCC_IMM12_BOTH]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
}

def SFPI_ENCC_MOD1_EU_R1 : I32EnumAttrCase<"EU_R1", 0, "EU_R1">;
def SFPI_ENCC_MOD1_EC_R1 : I32EnumAttrCase<"EC_R1", 1, "EC_R1">;
def SFPI_ENCC_MOD1_EI_R1 : I32EnumAttrCase<"EI_R1", 2, "EI_R1">;
def SFPI_ENCC_MOD1_EU_RI : I32EnumAttrCase<"EU_RI", 8, "EU_RI">;
def SFPI_ENCC_MOD1_EC_RI : I32EnumAttrCase<"EC_RI", 9, "EC_RI">;
def SFPI_ENCC_MOD1_EI_RI : I32EnumAttrCase<"EI_RI", 10, "EI_RI">;

def SFPI_ENCC_MOD1 : I32EnumAttr<"ENCC_MOD1", "SFPI Bool Mode",
                                 [SFPI_ENCC_MOD1_EU_R1, SFPI_ENCC_MOD1_EC_R1,
                                  SFPI_ENCC_MOD1_EI_R1, SFPI_ENCC_MOD1_EU_RI,
                                  SFPI_ENCC_MOD1_EC_RI, SFPI_ENCC_MOD1_EI_RI]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
}

def SFPI_PUSHC_MOD1_PUSH : I32EnumAttrCase<"PUSH", 0, "PUSH">;
def SFPI_PUSHC_MOD1_REPLACE : I32EnumAttrCase<"REPLACE", 1, "REPLACE">;

def SFPI_PUSHC_MOD1
    : I32EnumAttr<
          "PUSHC_MOD1",
          "SFPI Bool Mode", [SFPI_PUSHC_MOD1_PUSH, SFPI_PUSHC_MOD1_REPLACE]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
}

//===----------------------------------------------------------------------===//
// SFPI Leading Zero Count Mod1 Enums
//===----------------------------------------------------------------------===//

def SFPI_LZ_MOD1_CC_NONE : I32BitEnumAttrCase<"CC_NONE", 0, "CC_NONE">;
def SFPI_LZ_MOD1_CC_NE0 : I32BitEnumAttrCase<"CC_NE0", 2, "CC_NE0">;
def SFPI_LZ_MOD1_CC_COMP : I32BitEnumAttrCase<"CC_COMP", 8, "CC_COMP">;
def SFPI_LZ_MOD1_CC_EQ0 : I32BitEnumAttrCase<"CC_EQ0", 10, "CC_EQ0">;
def SFPI_LZ_MOD1_NOSGN_CC_NONE
    : I32BitEnumAttrCase<"NOSGN_CC_NONE", 4, "NOSGN_CC_NONE">;
def SFPI_LZ_MOD1_NOSGN_CC_NE0
    : I32BitEnumAttrCase<"NOSGN_CC_NE0", 6, "NOSGN_CC_NE0">;
def SFPI_LZ_MOD1_NOSGN_CC_COMP
    : I32BitEnumAttrCase<"NOSGN_CC_COMP", 12, "NOSGN_CC_COMP">;
def SFPI_LZ_MOD1_NOSGN_CC_EQ0
    : I32BitEnumAttrCase<"NOSGN_CC_EQ0", 14, "NOSGN_CC_EQ0">;

def SFPI_LZ_MOD1
    : I32BitEnumAttr<"LZ_MOD1", "SFPI Leading Zero Count Mod1 Mode",
                     [SFPI_LZ_MOD1_CC_NONE, SFPI_LZ_MOD1_CC_NE0,
                      SFPI_LZ_MOD1_CC_COMP, SFPI_LZ_MOD1_CC_EQ0,
                      SFPI_LZ_MOD1_NOSGN_CC_NONE, SFPI_LZ_MOD1_NOSGN_CC_NE0,
                      SFPI_LZ_MOD1_NOSGN_CC_COMP, SFPI_LZ_MOD1_NOSGN_CC_EQ0]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
  let separator = "|";
}

//===----------------------------------------------------------------------===//
// SFPI SDIVP2 Enums
//===----------------------------------------------------------------------===//

def SFPI_SDIVP2_MOD1_ADD : I32EnumAttrCase<"ADD", 0, "ADD">;

def SFPI_SDIVP2_MOD1
    : I32EnumAttr<"SDIVP2_MOD1",
                  "SFPI Div Power of 2", [SFPI_SDIVP2_MOD1_ADD]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
}

//===----------------------------------------------------------------------===//
// SFPI LUT Enums
//===----------------------------------------------------------------------===//

def SFPI_LUT_MOD0_SGN_UPDATE
    : I32BitEnumAttrCase<"SGN_UPDATE", 0, "SGN_UPDATE">;
def SFPI_LUT_MOD0_SGN_RETAIN
    : I32BitEnumAttrCase<"SGN_RETAIN", 4, "SGN_RETAIN">;

def SFPI_LUT_MOD0
    : I32BitEnumAttr<"LUT_MOD0", "SFPI LUT Mod0",
                     [SFPI_LUT_MOD0_SGN_UPDATE, SFPI_LUT_MOD0_SGN_RETAIN]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
  let separator = "|";
}

def SFPI_LUTFP32_MOD0_FP32_3ENTRY_TABLE
    : I32BitEnumAttrCase<"FP32_3ENTRY_TABLE", 0, "FP32_3ENTRY_TABLE">;
def SFPI_LUTFP32_MOD0_FP16_6ENTRY_TABLE1
    : I32BitEnumAttrCase<"FP16_6ENTRY_TABLE1", 2, "FP16_6ENTRY_TABLE1">;
def SFPI_LUTFP32_MOD0_FP16_6ENTRY_TABLE2
    : I32BitEnumAttrCase<"FP16_6ENTRY_TABLE2", 3, "FP16_6ENTRY_TABLE2">;
def SFPI_LUTFP32_MOD0_FP16_3ENTRY_TABLE
    : I32BitEnumAttrCase<"FP16_3ENTRY_TABLE", 10, "FP16_3ENTRY_TABLE">;
// def SFPI_LUTFP32_MOD0_SGN_UPDATE : I32BitEnumAttrCase<"SGN_UPDATE", 0,
// "SGN_UPDATE">;
def SFPI_LUTFP32_MOD0_SGN_RETAIN
    : I32BitEnumAttrCase<"SGN_RETAIN", 4, "SGN_RETAIN">;

def SFPI_LUTFP32_MOD0 : I32BitEnumAttr<"LUTFP32_MOD0", "SFPI LUT Mod0",
                                       [SFPI_LUTFP32_MOD0_FP32_3ENTRY_TABLE,
                                        SFPI_LUTFP32_MOD0_FP16_6ENTRY_TABLE1,
                                        SFPI_LUTFP32_MOD0_FP16_6ENTRY_TABLE2,
                                        SFPI_LUTFP32_MOD0_FP16_3ENTRY_TABLE,
                                        // SFPI_LUTFP32_MOD0_SGN_UPDATE,
                                        SFPI_LUTFP32_MOD0_SGN_RETAIN]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
  let separator = "|";
}

//===----------------------------------------------------------------------===//
// SFPI Cast Mod1 Enums
//===----------------------------------------------------------------------===//

def SFPI_CAST_MOD1_INT32_TO_FP32_RNE
    : I32EnumAttrCase<"INT32_TO_FP32_RNE", 0, "INT32_TO_FP32_RNE">;
def SFPI_CAST_MOD1_INT32_TO_FP32_RNS
    : I32EnumAttrCase<"INT32_TO_FP32_RNS", 1, "INT32_TO_FP32_RNS">;

def SFPI_CAST_MOD1 : I32EnumAttr<"CAST_MOD1", "SFPI Cast Mod1 Mode",
                                 [SFPI_CAST_MOD1_INT32_TO_FP32_RNE,
                                  SFPI_CAST_MOD1_INT32_TO_FP32_RNS]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
}

//===----------------------------------------------------------------------===//
// SFPI Stochastic Rounding Mode Enums
//===----------------------------------------------------------------------===//

def SFPI_STOCHRND_RND_EVEN : I32EnumAttrCase<"EVEN", 0, "EVEN">;
def SFPI_STOCHRND_RND_STOCH : I32EnumAttrCase<"STOCH", 1, "STOCH">;

def SFPI_STOCHRND_RND
    : I32EnumAttr<"STOCHRND_RND", "SFPI Stochastic Rounding Mode",
                  [SFPI_STOCHRND_RND_EVEN, SFPI_STOCHRND_RND_STOCH, ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
}

def SFPI_STOCHRND_MOD1_FP32_TO_FP16A
    : I32EnumAttrCase<"FP32_TO_FP16A", 0, "FP32_TO_FP16A">;
def SFPI_STOCHRND_MOD1_FP32_TO_FP16B
    : I32EnumAttrCase<"FP32_TO_FP16B", 1, "FP32_TO_FP16B">;
def SFPI_STOCHRND_MOD1_FP32_TO_UINT8
    : I32EnumAttrCase<"FP32_TO_UINT8", 2, "FP32_TO_UINT8">;
def SFPI_STOCHRND_MOD1_FP32_TO_INT8
    : I32EnumAttrCase<"FP32_TO_INT8", 3, "FP32_TO_INT8">;
def SFPI_STOCHRND_MOD1_INT32_TO_UINT8
    : I32EnumAttrCase<"INT32_TO_UINT8", 4, "INT32_TO_UINT8">;
def SFPI_STOCHRND_MOD1_INT32_TO_INT8
    : I32EnumAttrCase<"INT32_TO_INT8", 5, "INT32_TO_INT8">;
def SFPI_STOCHRND_MOD1_FP32_TO_UINT16
    : I32EnumAttrCase<"FP32_TO_UINT16", 6, "FP32_TO_UINT16">;
def SFPI_STOCHRND_MOD1_FP32_TO_INT16
    : I32EnumAttrCase<"FP32_TO_INT16", 7, "FP32_TO_INT16">;
def SFPI_STOCHRND_MOD1_IMM8 : I32EnumAttrCase<"IMM8", 8, "IMM8">;

def SFPI_STOCHRND_MOD1
    : I32EnumAttr<
          "STOCHRND_MOD1", "SFPI Stochastic Rounding Mode",
          [SFPI_STOCHRND_MOD1_FP32_TO_FP16A, SFPI_STOCHRND_MOD1_FP32_TO_FP16B,
           SFPI_STOCHRND_MOD1_FP32_TO_UINT8, SFPI_STOCHRND_MOD1_FP32_TO_INT8,
           SFPI_STOCHRND_MOD1_INT32_TO_UINT8, SFPI_STOCHRND_MOD1_INT32_TO_INT8,
           SFPI_STOCHRND_MOD1_FP32_TO_UINT16, SFPI_STOCHRND_MOD1_FP32_TO_INT16,
           SFPI_STOCHRND_MOD1_IMM8]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
}

//===----------------------------------------------------------------------===//
// SFPI Shft2 Mode Enums
//===----------------------------------------------------------------------===//

def SFPI_SHFT2_MOD1_COPY4 : I32EnumAttrCase<"COPY4", 0, "COPY4">;
def SFPI_SHFT2_MOD1_SUBVEC_CHAINED_COPY4
    : I32EnumAttrCase<"SUBVEC_CHAINED_COPY4", 1, "SUBVEC_CHAINED_COPY4">;
def SFPI_SHFT2_MOD1_SUBVEC_SHFLROR1_AND_COPY4
    : I32EnumAttrCase<"SUBVEC_SHFLROR1_AND_COPY4", 2,
                      "SUBVEC_SHFLROR1_AND_COPY4">;
def SFPI_SHFT2_MOD1_SUBVEC_SHFLROR1
    : I32EnumAttrCase<"SUBVEC_SHFLROR1", 3, "SUBVEC_SHFLROR1">;
def SFPI_SHFT2_MOD1_SUBVEC_SHFLSHR1
    : I32EnumAttrCase<"SUBVEC_SHFLSHR1", 4, "SUBVEC_SHFLSHR1">;
def SFPI_SHFT2_MOD1_SHFT_LREG : I32EnumAttrCase<"SHFT_LREG", 5, "SHFT_LREG">;
def SFPI_SHFT2_MOD1_SHFT_IMM : I32EnumAttrCase<"SHFT_IMM", 6, "SHFT_IMM">;

def SFPI_SHFT2_MOD1
    : I32EnumAttr<"SHFT2_MOD1", "SFPI Shft2 Mode",
                  [SFPI_SHFT2_MOD1_COPY4, SFPI_SHFT2_MOD1_SUBVEC_CHAINED_COPY4,
                   SFPI_SHFT2_MOD1_SUBVEC_SHFLROR1_AND_COPY4,
                   SFPI_SHFT2_MOD1_SUBVEC_SHFLROR1,
                   SFPI_SHFT2_MOD1_SUBVEC_SHFLSHR1, SFPI_SHFT2_MOD1_SHFT_LREG,
                   SFPI_SHFT2_MOD1_SHFT_IMM]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
}

//===----------------------------------------------------------------------===//
// SFPI Swap Mode Enums
//===----------------------------------------------------------------------===//

def SFPI_SWAP_MOD1_SWAP : I32EnumAttrCase<"SWAP", 0, "SWAP">;
def SFPI_SWAP_MOD1_VEC_MIN_MAX
    : I32EnumAttrCase<"VEC_MIN_MAX", 1, "VEC_MIN_MAX">;
def SFPI_SWAP_MOD1_SUBVEC_MIN01_MAX23
    : I32EnumAttrCase<"SUBVEC_MIN01_MAX23", 2, "SUBVEC_MIN01_MAX23">;
def SFPI_SWAP_MOD1_SUBVEC_MIN02_MAX13
    : I32EnumAttrCase<"SUBVEC_MIN02_MAX13", 3, "SUBVEC_MIN02_MAX13">;
def SFPI_SWAP_MOD1_SUBVEC_MIN03_MAX12
    : I32EnumAttrCase<"SUBVEC_MIN03_MAX12", 4, "SUBVEC_MIN03_MAX12">;
def SFPI_SWAP_MOD1_SUBVEC_MIN0_MAX123
    : I32EnumAttrCase<"SUBVEC_MIN0_MAX123", 5, "SUBVEC_MIN0_MAX123">;
def SFPI_SWAP_MOD1_SUBVEC_MIN1_MAX023
    : I32EnumAttrCase<"SUBVEC_MIN1_MAX023", 6, "SUBVEC_MIN1_MAX023">;
def SFPI_SWAP_MOD1_SUBVEC_MIN2_MAX013
    : I32EnumAttrCase<"SUBVEC_MIN2_MAX013", 7, "SUBVEC_MIN2_MAX013">;
def SFPI_SWAP_MOD1_SUBVEC_MIN3_MAX012
    : I32EnumAttrCase<"SUBVEC_MIN3_MAX012", 8, "SUBVEC_MIN3_MAX012">;

def SFPI_SWAP_MOD1
    : I32EnumAttr<
          "SWAP_MOD1", "SFPI Swap Mode",
          [SFPI_SWAP_MOD1_SWAP, SFPI_SWAP_MOD1_VEC_MIN_MAX,
           SFPI_SWAP_MOD1_SUBVEC_MIN01_MAX23, SFPI_SWAP_MOD1_SUBVEC_MIN02_MAX13,
           SFPI_SWAP_MOD1_SUBVEC_MIN03_MAX12, SFPI_SWAP_MOD1_SUBVEC_MIN0_MAX123,
           SFPI_SWAP_MOD1_SUBVEC_MIN1_MAX023, SFPI_SWAP_MOD1_SUBVEC_MIN2_MAX013,
           SFPI_SWAP_MOD1_SUBVEC_MIN3_MAX012]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
}

//===----------------------------------------------------------------------===//
// SFPI Configuration Destination Enums
//===----------------------------------------------------------------------===//

def SFPI_CONFIG_DEST_MACRO_INST0
    : I32EnumAttrCase<"MACRO_INST0", 0, "MACRO_INST0">;
def SFPI_CONFIG_DEST_MACRO_INST1
    : I32EnumAttrCase<"MACRO_INST1", 1, "MACRO_INST1">;
def SFPI_CONFIG_DEST_MACRO_INST2
    : I32EnumAttrCase<"MACRO_INST2", 2, "MACRO_INST2">;
def SFPI_CONFIG_DEST_MACRO_INST3
    : I32EnumAttrCase<"MACRO_INST3", 3, "MACRO_INST3">;
def SFPI_CONFIG_DEST_MACRO_SEQ0
    : I32EnumAttrCase<"MACRO_SEQ0", 4, "MACRO_SEQ0">;
def SFPI_CONFIG_DEST_MACRO_SEQ1
    : I32EnumAttrCase<"MACRO_SEQ1", 5, "MACRO_SEQ1">;
def SFPI_CONFIG_DEST_MACRO_SEQ2
    : I32EnumAttrCase<"MACRO_SEQ2", 6, "MACRO_SEQ2">;
def SFPI_CONFIG_DEST_MACRO_SEQ3
    : I32EnumAttrCase<"MACRO_SEQ3", 7, "MACRO_SEQ3">;
def SFPI_CONFIG_DEST_MACRO_CTRL
    : I32EnumAttrCase<"MACRO_CTRL", 8, "MACRO_CTRL">;
def SFPI_CONFIG_DEST_LREG11 : I32EnumAttrCase<"LREG11", 11, "LREG11">;
def SFPI_CONFIG_DEST_LREG12 : I32EnumAttrCase<"LREG12", 12, "LREG12">;
def SFPI_CONFIG_DEST_LREG13 : I32EnumAttrCase<"LREG13", 13, "LREG13">;
def SFPI_CONFIG_DEST_LREG14 : I32EnumAttrCase<"LREG14", 14, "LREG14">;
def SFPI_CONFIG_DEST_SFPU_CTRL : I32EnumAttrCase<"SFPU_CTRL", 15, "SFPU_CTRL">;

def SFPI_CONFIG_DEST
    : I32EnumAttr<"CONFIG_DEST", "SFPI Configuration Destination",
                  [SFPI_CONFIG_DEST_MACRO_INST0, SFPI_CONFIG_DEST_MACRO_INST1,
                   SFPI_CONFIG_DEST_MACRO_INST2, SFPI_CONFIG_DEST_MACRO_INST3,
                   SFPI_CONFIG_DEST_MACRO_SEQ0, SFPI_CONFIG_DEST_MACRO_SEQ1,
                   SFPI_CONFIG_DEST_MACRO_SEQ2, SFPI_CONFIG_DEST_MACRO_SEQ3,
                   SFPI_CONFIG_DEST_MACRO_CTRL, SFPI_CONFIG_DEST_LREG11,
                   SFPI_CONFIG_DEST_LREG12, SFPI_CONFIG_DEST_LREG13,
                   SFPI_CONFIG_DEST_LREG14, SFPI_CONFIG_DEST_SFPU_CTRL]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
}

def SFPI_CONFIG_MOD1_SRC_R0_LREG0
    : I32EnumAttrCase<"SRC_R0_LREG0", 0, "SRC_R0_LREG0">;

def SFPI_CONFIG_MOD1
    : I32EnumAttr<
          "CONFIG_MOD1",
          "SFPI Configuration Destination", [SFPI_CONFIG_MOD1_SRC_R0_LREG0]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
}

//===----------------------------------------------------------------------===//
// SFPI Constant Register Index Enums
//===----------------------------------------------------------------------===//

def SFPI_CREG_IDX_0P837300003
    : I32EnumAttrCase<"C0P837300003", 8, "C0P837300003">;
def SFPI_CREG_IDX_0 : I32EnumAttrCase<"C0", 9, "C0">;
def SFPI_CREG_IDX_1 : I32EnumAttrCase<"C1", 10, "C1">;
def SFPI_CREG_IDX_PRGM0 : I32EnumAttrCase<"PRGM0", 11, "PRGM0">;
def SFPI_CREG_IDX_PRGM1 : I32EnumAttrCase<"PRGM1", 12, "PRGM1">;
def SFPI_CREG_IDX_PRGM2 : I32EnumAttrCase<"PRGM2", 13, "PRGM2">;
def SFPI_CREG_IDX_PRGM3 : I32EnumAttrCase<"PRGM3", 14, "PRGM3">;
def SFPI_CREG_IDX_TILEID : I32EnumAttrCase<"TILEID", 15, "TILEID">;

def SFPI_CREG_IDX : I32EnumAttr<"CREG_IDX", "SFPI Constant Register Index",
                                [SFPI_CREG_IDX_0P837300003, SFPI_CREG_IDX_0,
                                 SFPI_CREG_IDX_1, SFPI_CREG_IDX_PRGM0,
                                 SFPI_CREG_IDX_PRGM1, SFPI_CREG_IDX_PRGM2,
                                 SFPI_CREG_IDX_PRGM3, SFPI_CREG_IDX_TILEID]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::sfpi";
}

#endif // TTMLIR_DIALECT_SFPI_IR_SFPIENUMS_TD
