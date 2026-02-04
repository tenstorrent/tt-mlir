// RUN: ttmlir-opt --split-input-file --ttir-to-ttmetal-fe-pipeline --ttir-to-ttmetal-me-pipeline --convert-d2m-to-ttkernel %s | FileCheck %s

!ttype_f32 = tensor<32x32xf32>
// CHECK-LABEL: func.func @test_where_f32
func.func @test_where_f32(%cond: !ttype_f32, %true_val: !ttype_f32, %false_val :!ttype_f32) -> (!ttype_f32) {
  // CHECK: ttkernel.where_tile_init() : () -> ()
  // CHECK: ttkernel.where_tile(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, <f32>)
  %0 = "ttir.where"(%cond, %true_val, %false_val) : (!ttype_f32, !ttype_f32, !ttype_f32) -> !ttype_f32
  return %0 : !ttype_f32
}

// -----

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_where_i32
func.func @test_where_i32(%cond: !ttype_i32, %true_val: !ttype_i32, %false_val :!ttype_i32) -> (!ttype_i32) {
  // CHECK: ttkernel.where_tile_init() : () -> ()
  // CHECK: ttkernel.where_tile(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, <si32>)
  %0 = "ttir.where"(%cond, %true_val, %false_val) : (!ttype_i32, !ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_bitwise_and_i32
func.func @test_bitwise_and_i32(%lhs: !ttype_i32, %rhs: !ttype_i32) -> (!ttype_i32) {
  // CHECK: ttkernel.binary_bitwise_tile_init() : () -> ()
  // CHECK: ttkernel.bitwise_and_binary_tile(%{{.*}}, %{{.*}}, %{{.*}}, <si32>)
  %0 = "ttir.bitwise_and"(%lhs, %rhs) : (!ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_bitwise_or_i32
func.func @test_bitwise_or_i32(%lhs: !ttype_i32, %rhs: !ttype_i32) -> (!ttype_i32) {
  // CHECK: ttkernel.binary_bitwise_tile_init() : () -> ()
  // CHECK: ttkernel.bitwise_or_binary_tile(%{{.*}}, %{{.*}}, %{{.*}}, <si32>)
  %0 = "ttir.bitwise_or"(%lhs, %rhs) : (!ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_bitwise_xor_i32
func.func @test_bitwise_xor_i32(%lhs: !ttype_i32, %rhs: !ttype_i32) -> (!ttype_i32) {
  // CHECK: ttkernel.binary_bitwise_tile_init() : () -> ()
  // CHECK: ttkernel.bitwise_xor_binary_tile(%{{.*}}, %{{.*}}, %{{.*}}, <si32>)
  %0 = "ttir.bitwise_xor"(%lhs, %rhs) : (!ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}
