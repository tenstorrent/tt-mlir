// RUN: ttmlir-opt %s -convert-sfpi-to-emitc | FileCheck %s

// Test basic SFPI operations conversion to EmitC builtin calls

// CHECK-LABEL: func.func @test_sfpi_add
func.func @test_sfpi_add(%arg0: vector<4x8xf32>, %arg1: vector<4x8xf32>) -> vector<4x8xf32> {
  // CHECK: %[[RESULT:.*]] = emitc.call "__builtin_rvtt_sfpadd"(%{{.*}}, %{{.*}}) : (__rvtt_vec_t, __rvtt_vec_t) -> __rvtt_vec_t
  %0 = sfpi.add %arg0, %arg1 : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// CHECK-LABEL: func.func @test_sfpi_mul
func.func @test_sfpi_mul(%arg0: vector<4x8xf32>, %arg1: vector<4x8xf32>) -> vector<4x8xf32> {
  // CHECK: %[[RESULT:.*]] = emitc.call "__builtin_rvtt_sfpmul"(%{{.*}}, %{{.*}}) : (__rvtt_vec_t, __rvtt_vec_t) -> __rvtt_vec_t
  %0 = sfpi.mul %arg0, %arg1 : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// CHECK-LABEL: func.func @test_sfpi_mad
func.func @test_sfpi_mad(%arg0: vector<4x8xf32>, %arg1: vector<4x8xf32>, %arg2: vector<4x8xf32>) -> vector<4x8xf32> {
  // CHECK: %[[RESULT:.*]] = emitc.call "__builtin_rvtt_sfpmad"(%{{.*}}, %{{.*}}, %{{.*}}) : (__rvtt_vec_t, __rvtt_vec_t, __rvtt_vec_t) -> __rvtt_vec_t  
  %0 = sfpi.mad %arg0, %arg1, %arg2 : vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// CHECK-LABEL: func.func @test_sfpi_mov
func.func @test_sfpi_mov(%arg0: vector<4x8xf32>) -> vector<4x8xf32> {
  // CHECK: %[[RESULT:.*]] = emitc.call "__builtin_rvtt_sfpmov"(%{{.*}}) : (__rvtt_vec_t) -> __rvtt_vec_t
  %0 = sfpi.mov %arg0 : vector<4x8xf32> -> vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// CHECK-LABEL: func.func @test_sfpi_abs
func.func @test_sfpi_abs(%arg0: vector<4x8xf32>) -> vector<4x8xf32> {
  // CHECK: %[[RESULT:.*]] = emitc.call "__builtin_rvtt_sfpabs"(%{{.*}}) : (__rvtt_vec_t) -> __rvtt_vec_t
  %0 = sfpi.abs %arg0 : vector<4x8xf32> -> vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// CHECK-LABEL: func.func @test_sfpi_bitwise
func.func @test_sfpi_bitwise(%arg0: vector<4x8xf32>, %arg1: vector<4x8xf32>) -> vector<4x8xf32> {
  // CHECK: %[[AND:.*]] = emitc.call "__builtin_rvtt_sfpand"(%{{.*}}, %{{.*}}) : (__rvtt_vec_t, __rvtt_vec_t) -> __rvtt_vec_t
  %0 = sfpi.and %arg0, %arg1 : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>
  
  // CHECK: %[[OR:.*]] = emitc.call "__builtin_rvtt_sfpor"(%{{.*}}, %{{.*}}) : (__rvtt_vec_t, __rvtt_vec_t) -> __rvtt_vec_t
  %1 = sfpi.or %0, %arg1 : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>
  
  // CHECK: %[[XOR:.*]] = emitc.call "__builtin_rvtt_sfpxor"(%{{.*}}, %{{.*}}) : (__rvtt_vec_t, __rvtt_vec_t) -> __rvtt_vec_t
  %2 = sfpi.xor %1, %arg0 : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>
  
  // CHECK: %[[NOT:.*]] = emitc.call "__builtin_rvtt_sfpnot"(%{{.*}}) : (__rvtt_vec_t) -> __rvtt_vec_t
  %3 = sfpi.not %2 : vector<4x8xf32> -> vector<4x8xf32>
  
  return %3 : vector<4x8xf32>
}

// CHECK-LABEL: func.func @test_sfpi_complex
func.func @test_sfpi_complex(%a: vector<4x8xf32>, %b: vector<4x8xf32>, %c: vector<4x8xf32>) -> vector<4x8xf32> {
  // CHECK: %[[ADD:.*]] = emitc.call "__builtin_rvtt_sfpadd"(%{{.*}}, %{{.*}}) : (__rvtt_vec_t, __rvtt_vec_t) -> __rvtt_vec_t
  %0 = sfpi.add %a, %b : vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>
  
  // CHECK: %[[MAD:.*]] = emitc.call "__builtin_rvtt_sfpmad"(%{{.*}}, %{{.*}}, %{{.*}}) : (__rvtt_vec_t, __rvtt_vec_t, __rvtt_vec_t) -> __rvtt_vec_t
  %1 = sfpi.mad %0, %b, %c : vector<4x8xf32>, vector<4x8xf32>, vector<4x8xf32> -> vector<4x8xf32>
  
  // CHECK: %[[ABS:.*]] = emitc.call "__builtin_rvtt_sfpabs"(%{{.*}}) : (__rvtt_vec_t) -> __rvtt_vec_t
  %2 = sfpi.abs %1 : vector<4x8xf32> -> vector<4x8xf32>
  
  return %2 : vector<4x8xf32>
}