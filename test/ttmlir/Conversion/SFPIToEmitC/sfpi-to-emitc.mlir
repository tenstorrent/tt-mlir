// RUN: ttmlir-opt %s -convert-sfpi-to-emitc -o %t
// RUN: FileCheck %s --input-file=%t
// RUN: ttmlir-translate -mlir-to-cpp %t -o %t.cpp
// RUN: %if sfpi %{ /opt/tenstorrent/sfpi/compiler/bin/riscv32-tt-elf-g++ -march=rv32im_xttwh -mabi=ilp32 -I /opt/tenstorrent/sfpi/include -c %t.cpp %}

// Test basic SFPI operations conversion to EmitC builtin calls

emitc.include <"wormhole/sfpi_hw.h">
emitc.include <"cstdint">
emitc.include <"tuple">
emitc.verbatim "namespace ckernel { void* instrn_buffer; }"

// CHECK-LABEL: func.func @test_sfpi_add
emitc.verbatim "inline"
func.func @test_sfpi_add(%arg0: vector<4x8xf32>, %arg1: vector<4x8xf32>) -> vector<4x8xf32> {
  // CHECK: %[[LITERAL:.*]] = emitc.literal "sfpi::SFPIADD_MOD1_ARG_LREG_DST" : i32
  // CHECK: %[[RESULT:.*]] = emitc.call_opaque "__builtin_rvtt_sfpadd"(%{{.*}}, %{{.*}}, %[[LITERAL]]) : (!emitc.opaque<"sfpi::__rvtt_vec_t">, !emitc.opaque<"sfpi::__rvtt_vec_t">, i32) -> !emitc.opaque<"sfpi::__rvtt_vec_t">
  %0 = sfpi.add %arg0, %arg1 {mod1 = #sfpi<add_mod1 ARG_LREG_DST>} : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// CHECK-LABEL: func.func @test_sfpi_mul
emitc.verbatim "inline"
func.func @test_sfpi_mul(%arg0: vector<4x8xf32>, %arg1: vector<4x8xf32>) -> vector<4x8xf32> {
  // CHECK: %[[LITERAL:.*]] = emitc.literal "sfpi::SFPMAD_MOD1_OFFSET_NONE" : i32
  // CHECK: %[[RESULT:.*]] = emitc.call_opaque "__builtin_rvtt_sfpmul"(%{{.*}}, %{{.*}}, %[[LITERAL]]) : (!emitc.opaque<"sfpi::__rvtt_vec_t">, !emitc.opaque<"sfpi::__rvtt_vec_t">, i32) -> !emitc.opaque<"sfpi::__rvtt_vec_t">
  %0 = sfpi.mul %arg0, %arg1 {mod1 = #sfpi<mad_mod1 OFFSET_NONE>} : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// CHECK-LABEL: func.func @test_sfpi_mad
emitc.verbatim "inline"
func.func @test_sfpi_mad(%arg0: vector<4x8xf32>, %arg1: vector<4x8xf32>, %arg2: vector<4x8xf32>) -> vector<4x8xf32> {
  // CHECK: %[[LITERAL:.*]] = emitc.literal "sfpi::SFPMAD_MOD1_OFFSET_NONE" : i32
  // CHECK: %[[RESULT:.*]] = emitc.call_opaque "__builtin_rvtt_sfpmad"(%{{.*}}, %{{.*}}, %{{.*}}, %[[LITERAL]]) : (!emitc.opaque<"sfpi::__rvtt_vec_t">, !emitc.opaque<"sfpi::__rvtt_vec_t">, !emitc.opaque<"sfpi::__rvtt_vec_t">, i32) -> !emitc.opaque<"sfpi::__rvtt_vec_t">
  %0 = sfpi.mad %arg0, %arg1, %arg2 {mod1 = #sfpi<mad_mod1 OFFSET_NONE>} : vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// CHECK-LABEL: func.func @test_sfpi_mov
emitc.verbatim "inline"
func.func @test_sfpi_mov(%arg0: vector<4x8xf32>) -> vector<4x8xf32> {
  // CHECK: %[[LITERAL:.*]] = emitc.literal "sfpi::SFPMOV_MOD1_COMPSIGN" : i32
  // CHECK: %[[RESULT:.*]] = emitc.call_opaque "__builtin_rvtt_sfpmov"(%{{.*}}, %[[LITERAL]]) : (!emitc.opaque<"sfpi::__rvtt_vec_t">, i32) -> !emitc.opaque<"sfpi::__rvtt_vec_t">
  %0 = sfpi.mov %arg0 {mod1 = #sfpi<mov_mod1 COMPSIGN>} : vector<4x8xf32> -> vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// CHECK-LABEL: func.func @test_sfpi_abs
emitc.verbatim "inline"
func.func @test_sfpi_abs(%arg0: vector<4x8xf32>) -> vector<4x8xf32> {
  // CHECK: %[[LITERAL:.*]] = emitc.literal "sfpi::SFPABS_MOD1_FLOAT" : i32
  // CHECK: %[[RESULT:.*]] = emitc.call_opaque "__builtin_rvtt_sfpabs"(%{{.*}}, %[[LITERAL]]) : (!emitc.opaque<"sfpi::__rvtt_vec_t">, i32) -> !emitc.opaque<"sfpi::__rvtt_vec_t">
  %0 = sfpi.abs %arg0 {mod1 = #sfpi<abs_mod1 FLOAT>} : vector<4x8xf32> -> vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// CHECK-LABEL: func.func @test_sfpi_bitwise
emitc.verbatim "inline"
func.func @test_sfpi_bitwise(%arg0: vector<4x8xf32>, %arg1: vector<4x8xf32>) -> vector<4x8xf32> {
  // CHECK: %[[AND:.*]] = emitc.call_opaque "__builtin_rvtt_sfpand"(%{{.*}}, %{{.*}}) : (!emitc.opaque<"sfpi::__rvtt_vec_t">, !emitc.opaque<"sfpi::__rvtt_vec_t">) -> !emitc.opaque<"sfpi::__rvtt_vec_t">
  %0 = sfpi.and %arg0, %arg1 : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: %[[OR:.*]] = emitc.call_opaque "__builtin_rvtt_sfpor"(%{{.*}}, %{{.*}}) : (!emitc.opaque<"sfpi::__rvtt_vec_t">, !emitc.opaque<"sfpi::__rvtt_vec_t">) -> !emitc.opaque<"sfpi::__rvtt_vec_t">
  %1 = sfpi.or %0, %arg1 : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: %[[XOR:.*]] = emitc.call_opaque "__builtin_rvtt_sfpxor"(%{{.*}}, %{{.*}}) : (!emitc.opaque<"sfpi::__rvtt_vec_t">, !emitc.opaque<"sfpi::__rvtt_vec_t">) -> !emitc.opaque<"sfpi::__rvtt_vec_t">
  %2 = sfpi.xor %1, %arg0 : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: %[[NOT:.*]] = emitc.call_opaque "__builtin_rvtt_sfpnot"(%{{.*}}) : (!emitc.opaque<"sfpi::__rvtt_vec_t">) -> !emitc.opaque<"sfpi::__rvtt_vec_t">
  %3 = sfpi.not %2 : vector<4x8xf32> -> vector<4x8xf32>

  return %3 : vector<4x8xf32>
}

// CHECK-LABEL: func.func @test_sfpi_complex
emitc.verbatim "inline"
func.func @test_sfpi_complex(%a: vector<4x8xf32>, %b: vector<4x8xf32>, %c: vector<4x8xf32>) -> vector<4x8xf32> {
  // CHECK: %[[ADD_LITERAL:.*]] = emitc.literal "sfpi::SFPIADD_MOD1_ARG_LREG_DST" : i32
  // CHECK: %[[ADD:.*]] = emitc.call_opaque "__builtin_rvtt_sfpadd"(%{{.*}}, %{{.*}}, %[[ADD_LITERAL]]) : (!emitc.opaque<"sfpi::__rvtt_vec_t">, !emitc.opaque<"sfpi::__rvtt_vec_t">, i32) -> !emitc.opaque<"sfpi::__rvtt_vec_t">
  %0 = sfpi.add %a, %b {mod1 = #sfpi<add_mod1 ARG_LREG_DST>} : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: %[[MAD_LITERAL:.*]] = emitc.literal "sfpi::SFPMAD_MOD1_OFFSET_NONE" : i32
  // CHECK: %[[MAD:.*]] = emitc.call_opaque "__builtin_rvtt_sfpmad"(%{{.*}}, %{{.*}}, %{{.*}}, %[[MAD_LITERAL]]) : (!emitc.opaque<"sfpi::__rvtt_vec_t">, !emitc.opaque<"sfpi::__rvtt_vec_t">, !emitc.opaque<"sfpi::__rvtt_vec_t">, i32) -> !emitc.opaque<"sfpi::__rvtt_vec_t">
  %1 = sfpi.mad %0, %b, %c {mod1 = #sfpi<mad_mod1 OFFSET_NONE>} : vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: %[[LITERAL:.*]] = emitc.literal "sfpi::SFPABS_MOD1_FLOAT" : i32
  // CHECK: %[[ABS:.*]] = emitc.call_opaque "__builtin_rvtt_sfpabs"(%{{.*}}, %[[LITERAL]]) : (!emitc.opaque<"sfpi::__rvtt_vec_t">, i32) -> !emitc.opaque<"sfpi::__rvtt_vec_t">
  %2 = sfpi.abs %1 {mod1 = #sfpi<abs_mod1 FLOAT>} : vector<4x8xf32> -> vector<4x8xf32>

  return %2 : vector<4x8xf32>
}

// Test FP Manipulation Immediate Operations
// CHECK-LABEL: func.func @test_sfpi_setfp_i
emitc.verbatim "inline"
func.func @test_sfpi_setfp_i(%arg0: vector<4x8xf32>, %imm: i32) -> (vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32>) {
  // CHECK: %[[SETEXP:.*]] = emitc.call_opaque "__builtin_rvtt_sfpsetexp_i"(%{{.*}}, %{{.*}}) : (i32, !emitc.opaque<"sfpi::__rvtt_vec_t">) -> !emitc.opaque<"sfpi::__rvtt_vec_t">
  %0 = sfpi.setexp_i %imm, %arg0 : i32, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: %[[SETMAN:.*]] = emitc.call_opaque "__builtin_rvtt_sfpsetman_i"(%{{.*}}, %{{.*}} : (i32, !emitc.opaque<"sfpi::__rvtt_vec_t">, i32) -> !emitc.opaque<"sfpi::__rvtt_vec_t">
  %1 = sfpi.setman_i %imm, %arg0 : i32, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: %[[SETSGN:.*]] = emitc.call_opaque "__builtin_rvtt_sfpsetsgn_i"(%{{.*}}, %{{.*}}) : (i32, !emitc.opaque<"sfpi::__rvtt_vec_t">) -> !emitc.opaque<"sfpi::__rvtt_vec_t">
  %2 = sfpi.setsgn_i %imm, %arg0 : i32, vector<4x8xf32> -> vector<4x8xf32>

  return %0, %1, %2 : vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32>
}

// Test Integer Arithmetic Operations
// CHECK-LABEL: func.func @test_sfpi_xiadd
emitc.verbatim "inline"
func.func @test_sfpi_xiadd(%arg0: vector<4x8xi32>, %arg1: vector<4x8xi32>, %imm: i32) -> (vector<4x8xi32>, vector<4x8xi32>) {
  // CHECK: %[[IMMADD_LITERAL:.*]] = emitc.literal "sfpi::SFPXIADD_MOD1_SIGNED" : i32
  // CHECK: %[[IMMADD:.*]] = emitc.call_opaque "__builtin_rvtt_sfpxiadd_i"(%{{.*}}, %{{.*}}, %[[IMMADD_LITERAL]]) : (!emitc.opaque<"sfpi::__rvtt_vec_t">, i32, i32) -> !emitc.opaque<"sfpi::__rvtt_vec_t">
  %0 = sfpi.xiadd_i %arg0, %imm {mod1 = #sfpi<xiadd_mod1 SIGNED>} : vector<4x8xi32>, i32 -> vector<4x8xi32>

  // CHECK: %[[VECADD_LITERAL:.*]] = emitc.literal "sfpi::SFPXIADD_MOD1_IS_SUB" : i32
  // CHECK: %[[VECADD:.*]] = emitc.call_opaque "__builtin_rvtt_sfpxiadd_v"(%{{.*}}, %{{.*}}, %[[VECADD_LITERAL]]) : (!emitc.opaque<"sfpi::__rvtt_vec_t">, !emitc.opaque<"sfpi::__rvtt_vec_t">, i32) -> !emitc.opaque<"sfpi::__rvtt_vec_t">
  %1 = sfpi.xiadd_v %arg0, %arg1 {mod1 = #sfpi<xiadd_mod1 IS_SUB>} : vector<4x8xi32>, vector<4x8xi32> -> vector<4x8xi32>

  return %0, %1 : vector<4x8xi32>, vector<4x8xi32>
}

// Test Extended LUT Operations
// CHECK-LABEL: func.func @test_sfpi_lut_extended
emitc.verbatim "inline"
func.func @test_sfpi_lut_extended(%l0: vector<4x8xf32>, %l1: vector<4x8xf32>, %l2: vector<4x8xf32>, %l3: vector<4x8xf32>, %l4: vector<4x8xf32>, %l5: vector<4x8xf32>, %l6: vector<4x8xf32>) -> (vector<4x8xf32>, vector<4x8xf32>) {
  // CHECK: %[[LUT_FP32_3R_LITERAL:.*]] = emitc.literal "sfpi::SFPLUTFP32_MOD0_FP32_3ENTRY_TABLE" : i32
  // CHECK: %[[LUT_FP32_3R:.*]] = emitc.call_opaque "__builtin_rvtt_sfplutfp32_3r"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[LUT_FP32_3R_LITERAL]]) : (!emitc.opaque<"sfpi::__rvtt_vec_t">, !emitc.opaque<"sfpi::__rvtt_vec_t">, !emitc.opaque<"sfpi::__rvtt_vec_t">, !emitc.opaque<"sfpi::__rvtt_vec_t">, i32) -> !emitc.opaque<"sfpi::__rvtt_vec_t">
  %1 = sfpi.lutfp32_3r %l0, %l1, %l2, %l3 {mod0 = #sfpi<lutfp32_mod0 FP32_3ENTRY_TABLE>} : vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>

  // CHECK: %[[LUT_FP32_6R_LITERAL:.*]] = emitc.literal "sfpi::SFPLUTFP32_MOD0_FP16_6ENTRY_TABLE1" : i32
  // CHECK: %[[LUT_FP32_6R:.*]] = emitc.call_opaque "__builtin_rvtt_sfplutfp32_6r"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[LUT_FP32_6R_LITERAL]]) : (!emitc.opaque<"sfpi::__rvtt_vec_t">, !emitc.opaque<"sfpi::__rvtt_vec_t">, !emitc.opaque<"sfpi::__rvtt_vec_t">, !emitc.opaque<"sfpi::__rvtt_vec_t">, !emitc.opaque<"sfpi::__rvtt_vec_t">, !emitc.opaque<"sfpi::__rvtt_vec_t">, !emitc.opaque<"sfpi::__rvtt_vec_t">, i32) -> !emitc.opaque<"sfpi::__rvtt_vec_t">
  %2 = sfpi.lutfp32_6r %l0, %l1, %l2, %l4, %l5, %l6, %l3 {mod0 = #sfpi<lutfp32_mod0 FP16_6ENTRY_TABLE1>} : vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>

  return %1, %2 : vector<4x8xf32>, vector<4x8xf32>
}

// Test Load and Store Operations
// CHECK-LABEL: func.func @test_sfpi_load_store
emitc.verbatim "inline"
func.func @test_sfpi_load_store(%addr: i32, %data: vector<4x8xf32>) -> vector<4x8xf32> {
  // CHECK: %[[LOAD_FMT_LITERAL:.*]] = emitc.literal "sfpi::SFPLOAD_MOD0_FMT_FP32" : i32
  // CHECK: %[[LOAD_MODE_LITERAL:.*]] = emitc.literal "sfpi::SFPLOAD_ADDR_MODE_NOINC" : i32
  // CHECK: %[[LOAD:.*]] = emitc.call_opaque "__builtin_rvtt_sfpload"(%[[LOAD_FMT_LITERAL]], %[[LOAD_MODE_LITERAL]], %{{.*}}) : (i32, i32, i32) -> !emitc.opaque<"sfpi::__rvtt_vec_t">
  %0 = sfpi.load %addr {mod0 = #sfpi<load_mod0 FMT_FP32>, mode = #sfpi<load_addr_mode NOINC>} : i32 -> vector<4x8xf32>

  // CHECK: %[[STORE_FMT_LITERAL:.*]] = emitc.literal "sfpi::SFPSTORE_MOD0_FMT_FP16A" : i32
  // CHECK: %[[STORE_MODE_LITERAL:.*]] = emitc.literal "sfpi::SFPSTORE_ADDR_MODE_NOINC" : i32
  // CHECK: emitc.call_opaque "__builtin_rvtt_sfpstore"(%{{.*}}, %[[STORE_FMT_LITERAL]], %[[STORE_MODE_LITERAL]], %{{.*}}) : (!emitc.opaque<"sfpi::__rvtt_vec_t">, i32, i32, i32) -> ()
  sfpi.store %0, %addr {mod0 = #sfpi<store_mod0 FMT_FP16A>, mode = #sfpi<store_addr_mode NOINC>} : vector<4x8xf32>, i32

  return %data : vector<4x8xf32>
}

// Test Advanced Load Immediate Operations
// CHECK-LABEL: func.func @test_sfpi_xloadi
emitc.verbatim "inline"
func.func @test_sfpi_xloadi(%imm16: i32) -> vector<4x8xf32> {
  // CHECK: %[[XLOADI_LITERAL:.*]] = emitc.literal "sfpi::SFPXLOADI_MOD0_FLOAT" : i32
  // CHECK: %[[XLOADI:.*]] = emitc.call_opaque "__builtin_rvtt_sfpxloadi"(%[[XLOADI_LITERAL]], %{{.*}}) : (i32, i32) -> !emitc.opaque<"sfpi::__rvtt_vec_t">
  %0 = sfpi.xloadi %imm16 {mod0 = #sfpi<xloadi_mod0 FLOAT>} : i32 -> vector<4x8xf32>

  return %0 : vector<4x8xf32>
}

// Test Load with Different Formats
// CHECK-LABEL: func.func @test_sfpi_load_formats
emitc.verbatim "inline"
func.func @test_sfpi_load_formats(%addr: i32) -> (vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32>) {
  // CHECK: %[[LOAD1_FMT_LITERAL:.*]] = emitc.literal "sfpi::SFPLOAD_MOD0_FMT_FP16A" : i32
  // CHECK: %[[LOAD1_MODE_LITERAL:.*]] = emitc.literal "sfpi::SFPLOAD_ADDR_MODE_NOINC" : i32
  // CHECK: %[[LOAD1:.*]] = emitc.call_opaque "__builtin_rvtt_sfpload"(%[[LOAD1_FMT_LITERAL]], %[[LOAD1_MODE_LITERAL]], %{{.*}}) : (i32, i32, i32) -> !emitc.opaque<"sfpi::__rvtt_vec_t">
  %0 = sfpi.load %addr {mod0 = #sfpi<load_mod0 FMT_FP16A>, mode = #sfpi<load_addr_mode NOINC>} : i32 -> vector<4x8xf32>

  // CHECK: %[[LOAD2_FMT_LITERAL:.*]] = emitc.literal "sfpi::SFPLOAD_MOD0_FMT_FP16B" : i32
  // CHECK: %[[LOAD2_MODE_LITERAL:.*]] = emitc.literal "sfpi::SFPLOAD_ADDR_MODE_NOINC" : i32
  // CHECK: %[[LOAD2:.*]] = emitc.call_opaque "__builtin_rvtt_sfpload"(%[[LOAD2_FMT_LITERAL]], %[[LOAD2_MODE_LITERAL]], %{{.*}}) : (i32, i32, i32) -> !emitc.opaque<"sfpi::__rvtt_vec_t">
  %1 = sfpi.load %addr {mod0 = #sfpi<load_mod0 FMT_FP16B>, mode = #sfpi<load_addr_mode NOINC>} : i32 -> vector<4x8xf32>

  // CHECK: %[[LOAD3_FMT_LITERAL:.*]] = emitc.literal "sfpi::SFPLOAD_MOD0_FMT_SRCB" : i32
  // CHECK: %[[LOAD3_MODE_LITERAL:.*]] = emitc.literal "sfpi::SFPLOAD_ADDR_MODE_NOINC" : i32
  // CHECK: %[[LOAD3:.*]] = emitc.call_opaque "__builtin_rvtt_sfpload"(%[[LOAD3_FMT_LITERAL]], %[[LOAD3_MODE_LITERAL]], %{{.*}}) : (i32, i32, i32) -> !emitc.opaque<"sfpi::__rvtt_vec_t">
  %2 = sfpi.load %addr {mod0 = #sfpi<load_mod0 FMT_SRCB>, mode = #sfpi<load_addr_mode NOINC>} : i32 -> vector<4x8xf32>

  return %0, %1, %2 : vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32>
}

// Test Store with Different Formats
// CHECK-LABEL: func.func @test_sfpi_store_formats
emitc.verbatim "inline"
func.func @test_sfpi_store_formats(%addr: i32, %data1: vector<4x8xf32>, %data3: vector<4x8xf32>) {
  // CHECK: %[[STORE1_FMT_LITERAL:.*]] = emitc.literal "sfpi::SFPSTORE_MOD0_FMT_FP16B" : i32
  // CHECK: %[[STORE1_MODE_LITERAL:.*]] = emitc.literal "sfpi::SFPSTORE_ADDR_MODE_NOINC" : i32
  // CHECK: emitc.call_opaque "__builtin_rvtt_sfpstore"(%{{.*}}, %[[STORE1_FMT_LITERAL]], %[[STORE1_MODE_LITERAL]], %{{.*}}) : (!emitc.opaque<"sfpi::__rvtt_vec_t">, i32, i32, i32) -> ()
  sfpi.store %data1, %addr {mod0 = #sfpi<store_mod0 FMT_FP16B>, mode = #sfpi<store_addr_mode NOINC>} : vector<4x8xf32>, i32

  // CHECK: %[[STORE3_FMT_LITERAL:.*]] = emitc.literal "sfpi::SFPSTORE_MOD0_FMT_INT32_TO_SM" : i32
  // CHECK: %[[STORE3_MODE_LITERAL:.*]] = emitc.literal "sfpi::SFPSTORE_ADDR_MODE_NOINC" : i32
  // CHECK: emitc.call_opaque "__builtin_rvtt_sfpstore"(%{{.*}}, %[[STORE3_FMT_LITERAL]], %[[STORE3_MODE_LITERAL]], %{{.*}}) : (!emitc.opaque<"sfpi::__rvtt_vec_t">, i32, i32, i32) -> ()
  sfpi.store %data3, %addr {mod0 = #sfpi<store_mod0 FMT_INT32_TO_SM>, mode = #sfpi<store_addr_mode NOINC>} : vector<4x8xf32>, i32

  return
}
