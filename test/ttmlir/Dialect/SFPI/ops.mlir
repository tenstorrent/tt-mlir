// RUN: ttmlir-opt %s --verify-diagnostics | FileCheck %s

// Unit tests for all SFPI dialect operations

func.func @test_sfpi_ops(%vec_f32: vector<4x8xf32>, %vec_i32: vector<4x8xi32>, %addr: i32, %imm_i32: i32, %imm_f32: f32, %imm_i1: i1) {

  //===----------------------------------------------------------------------===//
  // Data Movement Operations
  //===----------------------------------------------------------------------===//

  // CHECK: sfpi.load
  %load_result = sfpi.load %addr {mod0 = #sfpi<load_mod0 FMT_FP32>, mode = #sfpi<load_addr_mode NOINC>} : i32 -> vector<4x8xf32>

  // CHECK: sfpi.store
  sfpi.store %vec_f32, %addr {mod0 = #sfpi<store_mod0 FMT_FP32>, mode = #sfpi<store_addr_mode NOINC>} : vector<4x8xf32>, i32

  // CHECK: sfpi.xloadi
  %xloadi_result = sfpi.xloadi %imm_i32 {mod0 = #sfpi<xloadi_mod0 FLOAT>} : i32 -> vector<4x8xf32>

  // CHECK: sfpi.mov
  %mov_result = sfpi.mov %vec_f32 {mod1 = #sfpi<mov_mod1 COMPSIGN>} : vector<4x8xf32> -> vector<4x8xf32>

  //===----------------------------------------------------------------------===//
  // Arithmetic Operations
  //===----------------------------------------------------------------------===//

  // CHECK: sfpi.add
  %add_result = sfpi.add %vec_f32, %vec_f32 {mod1 = #sfpi<add_mod1 ARG_LREG_DST>} : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: sfpi.mul
  %mul_result = sfpi.mul %vec_f32, %vec_f32 {mod1 = #sfpi<mad_mod1 OFFSET_NONE>} : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: sfpi.mad
  %mad_result = sfpi.mad %vec_f32, %vec_f32, %vec_f32 {mod1 = #sfpi<mad_mod1 OFFSET_NONE>} : vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: sfpi.divp2
  %divp2_result = sfpi.divp2 %vec_f32, %imm_i32 : vector<4x8xf32>, i32 -> vector<4x8xf32>

  // CHECK: sfpi.xiadd_i
  %xiadd_i_result = sfpi.xiadd_i %vec_i32, %imm_i32 {mod1 = #sfpi<xiadd_mod1 SIGNED>} : vector<4x8xi32>, i32 -> vector<4x8xi32>

  // CHECK: sfpi.xiadd_v
  %xiadd_v_result = sfpi.xiadd_v %vec_i32, %vec_i32 {mod1 = #sfpi<xiadd_mod1 IS_SUB>} : vector<4x8xi32>, vector<4x8xi32> -> vector<4x8xi32>

  //===----------------------------------------------------------------------===//
  // Bitwise Operations
  //===----------------------------------------------------------------------===//

  // CHECK: sfpi.and
  %and_result = sfpi.and %vec_f32, %vec_f32 : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: sfpi.or
  %or_result = sfpi.or %vec_f32, %vec_f32 : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: sfpi.xor
  %xor_result = sfpi.xor %vec_f32, %vec_f32 : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: sfpi.not
  %not_result = sfpi.not %vec_f32 : vector<4x8xf32> -> vector<4x8xf32>

  //===----------------------------------------------------------------------===//
  // Floating-Point Manipulation Operations
  //===----------------------------------------------------------------------===//

  // CHECK: sfpi.setexp_i
  %setexp_i_result = sfpi.setexp_i %imm_i32, %vec_f32 : i32, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: sfpi.setexp_v
  %setexp_v_result = sfpi.setexp_v %imm_i32, %vec_f32 : i32, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: sfpi.setman_i
  %setman_i_result = sfpi.setman_i %imm_i32, %vec_f32 : i32, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: sfpi.setman_v
  %setman_v_result = sfpi.setman_v %vec_f32, %vec_i32 : vector<4x8xf32>, vector<4x8xi32> -> vector<4x8xf32>

  // CHECK: sfpi.setsgn_i
  %setsgn_i_result = sfpi.setsgn_i %imm_i32, %vec_f32 : i32, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: sfpi.setsgn_v
  %setsgn_v_result = sfpi.setsgn_v %vec_f32, %vec_i32 : vector<4x8xf32>, vector<4x8xi32> -> vector<4x8xf32>

  // CHECK: sfpi.exexp
  %exexp_result = sfpi.exexp %vec_f32 {mod1 = #sfpi<exexp_mod1 DEBIAS>} : vector<4x8xf32> -> vector<4x8xi32>

  // CHECK: sfpi.exman
  %exman_result = sfpi.exman %vec_f32 {mod1 = #sfpi<exman_mod1 PAD8>} : vector<4x8xf32> -> vector<4x8xi32>

  // CHECK: sfpi.abs
  %abs_result = sfpi.abs %vec_f32 {mod1 = #sfpi<abs_mod1 FLOAT>} : vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: sfpi.lz
  %lz_result = sfpi.lz %vec_f32 {mod1 = #sfpi<lz_mod1 CC_NONE>} : vector<4x8xf32> -> vector<4x8xi32>

  //===----------------------------------------------------------------------===//
  // Comparison Operations
  //===----------------------------------------------------------------------===//

  // CHECK: sfpi.xfcmps
  %xfcmps_result = sfpi.xfcmps %vec_f32, %vec_f32 {mod1 = #sfpi<xscmp_mod1 FMT_FLOAT>} : vector<4x8xf32>, vector<4x8xf32> -> i1

  // CHECK: sfpi.xfcmpv
  %xfcmpv_result = sfpi.xfcmpv %vec_f32, %vec_f32 {mod1 = #sfpi<xcmp_mod1 CC_EQ>} : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xi32>

  // CHECK: sfpi.xicmps
  %xicmps_result = sfpi.xicmps %vec_i32, %imm_i32 {mod1 = #sfpi<xscmp_mod1 FMT_FLOAT>} : vector<4x8xi32>, i32 -> i1

  // CHECK: sfpi.xicmpv
  %xicmpv_result = sfpi.xicmpv %vec_i32, %vec_i32 {mod1 = #sfpi<xcmp_mod1 CC_EQ>} : vector<4x8xi32>, vector<4x8xi32> -> vector<4x8xi32>

  //===----------------------------------------------------------------------===//
  // Type Conversion Operations
  //===----------------------------------------------------------------------===//

  // CHECK: sfpi.cast
  %cast_result = sfpi.cast %vec_f32 {mod1 = #sfpi<cast_mod1 INT32_TO_FP32_RNE>} : vector<4x8xf32> -> vector<4x8xf32>

  //===----------------------------------------------------------------------===//
  // Specialized Operations
  //===----------------------------------------------------------------------===//

  // CHECK: sfpi.shft_i
  %shft_i_result = sfpi.shft_i %vec_f32, %imm_i32 : vector<4x8xf32>, i32 -> vector<4x8xf32>

  // CHECK: sfpi.shft_v
  %shft_v_result = sfpi.shft_v %vec_f32, %vec_f32 : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: sfpi.shft2_i
  %shft2_i_result = sfpi.shft2_i %vec_f32, %imm_i32 : vector<4x8xf32>, i32 -> vector<4x8xf32>

  // CHECK: sfpi.shft2_v
  %shft2_v_result = sfpi.shft2_v %vec_f32, %vec_f32 : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: sfpi.shft2_g
  sfpi.shft2_g %vec_f32, %vec_f32, %vec_f32, %vec_f32 {mod1 = #sfpi<shft2_mod1 COPY4>} : vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32>

  // CHECK: sfpi.shft2_ge
  sfpi.shft2_ge %vec_f32, %vec_f32, %vec_f32, %vec_f32, %vec_f32 : vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32>

  // CHECK: sfpi.shft2_e
  %shft2_e_result = sfpi.shft2_e %vec_f32 {mod1 = #sfpi<shft2_mod1 COPY4>} : vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: sfpi.stochrnd_i
  %stochrnd_i_result = sfpi.stochrnd_i %imm_i32, %vec_f32 {mode = #sfpi<stoch_rnd_rnd STOCH>, mod1 = #sfpi<stoch_rnd_mod1 FP32_TO_FP16B>} : i32, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: sfpi.stochrnd_v
  %stochrnd_v_result = sfpi.stochrnd_v %imm_i32, %vec_f32 {mode = #sfpi<stoch_rnd_rnd STOCH>, mod1 = #sfpi<stoch_rnd_mod1 FP32_TO_FP16B>} : i32, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: sfpi.lut
  %lut_result = sfpi.lut %vec_f32, %vec_f32, %vec_f32, %vec_f32 {mod0 = #sfpi<lut_mod0 SGN_UPDATE>} : vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: sfpi.lutfp32_3r
  %lutfp32_3r_result = sfpi.lutfp32_3r %vec_f32, %vec_f32, %vec_f32, %vec_f32 {mod0 = #sfpi<lutfp32_mod0 FP32_3ENTRY_TABLE>} : vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: sfpi.lutfp32_6r
  %lutfp32_6r_result = sfpi.lutfp32_6r %vec_f32, %vec_f32, %vec_f32, %vec_f32, %vec_f32, %vec_f32, %vec_f32 {mod0 = #sfpi<lutfp32_mod0 FP16_6ENTRY_TABLE1>} : vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: sfpi.swap
  %swap_result = sfpi.swap %vec_f32, %vec_f32 {mod1 = #sfpi<swap_mod1 SWAP>} : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: sfpi.transp
  sfpi.transp %vec_f32, %vec_f32, %vec_f32, %vec_f32 : vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32>

  //===----------------------------------------------------------------------===//
  // Condition Code Operations
  //===----------------------------------------------------------------------===//

  // CHECK: sfpi.setcc_i
  sfpi.setcc_i %imm_i32 {mod1 = #sfpi<setcc_mod1 LREG_LT0>} : i32

  // CHECK: sfpi.setcc_v
  sfpi.setcc_v %vec_f32 {mod1 = #sfpi<setcc_mod1 LREG_LT0>} : vector<4x8xf32>

  // CHECK: sfpi.encc
  sfpi.encc %imm_i32 {mod1 = #sfpi<encc_mod1 EU_R1>} : i32

  // CHECK: sfpi.pushc
  sfpi.pushc

  // CHECK: sfpi.popc
  sfpi.popc

  // CHECK: sfpi.compc
  sfpi.compc

  //===----------------------------------------------------------------------===//
  // Boolean and Conditional Operations
  //===----------------------------------------------------------------------===//

  // CHECK: sfpi.xbool
  %xbool_result = sfpi.xbool %imm_i32, %vec_f32, %vec_f32 : i32, vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: sfpi.xcondb
  sfpi.xcondb %vec_f32, %imm_i32 : vector<4x8xf32>, i32

  // CHECK: sfpi.xcondi
  sfpi.xcondi %imm_i32 : i32

  // CHECK: sfpi.xvif
  sfpi.xvif

  //===----------------------------------------------------------------------===//
  // Register Management Operations
  //===----------------------------------------------------------------------===//

  // CHECK: sfpi.assignlreg
  sfpi.assignlreg %imm_i32 : i32

  // CHECK: sfpi.assign_lv
  %assign_lv_result = sfpi.assign_lv %vec_f32, %imm_i32 : vector<4x8xf32>, i32 -> vector<4x8xf32>

  // CHECK: sfpi.preservelreg
  sfpi.preservelreg %imm_i32, %imm_i32 : i32, i32

  //===----------------------------------------------------------------------===//
  // Configuration Operations
  //===----------------------------------------------------------------------===//

  // CHECK: sfpi.config_v
  sfpi.config_v %imm_i32 {dest = #sfpi<config_dest SFPU_CTRL>} : i32

  return
}
