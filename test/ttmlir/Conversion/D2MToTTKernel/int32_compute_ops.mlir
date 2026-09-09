// RUN: ttmlir-opt --split-input-file --d2m-fe-pipeline --d2m-be-pipeline --convert-d2m-to-ttkernel %s | FileCheck %s

// Unary SFPU int32 ops.

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_abs_i32
func.func @test_abs_i32(%in: !ttype_i32) -> (!ttype_i32) {
  // CHECK: ttkernel.abs_tile_init
  // CHECK: ttkernel.abs_tile_int32(
  %0 = "ttir.abs"(%in) : (!ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_neg_i32
func.func @test_neg_i32(%in: !ttype_i32) -> (!ttype_i32) {
  // CHECK: ttkernel.negative_tile_init
  // CHECK: ttkernel.negative_tile_int32(
  %0 = "ttir.neg"(%in) : (!ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_relu_i32
func.func @test_relu_i32(%in: !ttype_i32) -> (!ttype_i32) {
  // CHECK: ttkernel.relu_tile_init
  // CHECK: ttkernel.relu_tile_int32(
  %0 = "ttir.relu"(%in) : (!ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

// Binary int32 comparison ops. The SFPU `*_binary_tile` API only writes fp32
// 1.0/0.0 into dst, which would be reinterpreted as garbage int32 bits if
// used for integer outputs. TTIRToD2M therefore decomposes integer
// comparisons into (a - b) followed by a compare-with-zero op, and
// D2MToTTKernel selects the int32 SFPU variants via IntComputeOpMap
// (`sub_int_tile` and `*z_tile_int32`).

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_eq_i32
func.func @test_eq_i32(%lhs: !ttype_i32, %rhs: !ttype_i32) -> (!ttype_i32) {
  // CHECK: ttkernel.sub_int_tile_init
  // CHECK: ttkernel.sub_int_tile({{.*}}, <si32>)
  // CHECK: ttkernel.eqz_tile_init
  // CHECK: ttkernel.eqz_tile_int32(
  %0 = "ttir.eq"(%lhs, %rhs) : (!ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_ne_i32
func.func @test_ne_i32(%lhs: !ttype_i32, %rhs: !ttype_i32) -> (!ttype_i32) {
  // CHECK: ttkernel.sub_int_tile_init
  // CHECK: ttkernel.sub_int_tile({{.*}}, <si32>)
  // CHECK: ttkernel.nez_tile_init
  // CHECK: ttkernel.nez_tile_int32(
  %0 = "ttir.ne"(%lhs, %rhs) : (!ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_gt_i32
func.func @test_gt_i32(%lhs: !ttype_i32, %rhs: !ttype_i32) -> (!ttype_i32) {
  // CHECK: ttkernel.sub_int_tile_init
  // CHECK: ttkernel.sub_int_tile({{.*}}, <si32>)
  // CHECK: ttkernel.gtz_tile_init
  // CHECK: ttkernel.gtz_tile_int32(
  %0 = "ttir.gt"(%lhs, %rhs) : (!ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_ge_i32
func.func @test_ge_i32(%lhs: !ttype_i32, %rhs: !ttype_i32) -> (!ttype_i32) {
  // CHECK: ttkernel.sub_int_tile_init
  // CHECK: ttkernel.sub_int_tile({{.*}}, <si32>)
  // CHECK: ttkernel.gez_tile_init
  // CHECK: ttkernel.gez_tile_int32(
  %0 = "ttir.ge"(%lhs, %rhs) : (!ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_lt_i32
func.func @test_lt_i32(%lhs: !ttype_i32, %rhs: !ttype_i32) -> (!ttype_i32) {
  // ttir.lt(a, b) is canonicalized to ttir.gt(b, a); for integer operands
  // the result then decomposes to (b - a) followed by gtz_tile_int32.
  // CHECK: ttkernel.sub_int_tile_init
  // CHECK: ttkernel.sub_int_tile({{.*}}, <si32>)
  // CHECK: ttkernel.gtz_tile_init
  // CHECK: ttkernel.gtz_tile_int32(
  %0 = "ttir.lt"(%lhs, %rhs) : (!ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_le_i32
func.func @test_le_i32(%lhs: !ttype_i32, %rhs: !ttype_i32) -> (!ttype_i32) {
  // ttir.le(a, b) is canonicalized to ttir.ge(b, a); for integer operands
  // the result then decomposes to (b - a) followed by gez_tile_int32.
  // CHECK: ttkernel.sub_int_tile_init
  // CHECK: ttkernel.sub_int_tile({{.*}}, <si32>)
  // CHECK: ttkernel.gez_tile_init
  // CHECK: ttkernel.gez_tile_int32(
  %0 = "ttir.le"(%lhs, %rhs) : (!ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

// Binary int32 ops.

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_add_i32
func.func @test_add_i32(%lhs: !ttype_i32, %rhs: !ttype_i32) -> (!ttype_i32) {
  // CHECK: ttkernel.add_int_tile_init
  // CHECK: ttkernel.add_int_tile({{.*}}, <si32>)
  %0 = "ttir.add"(%lhs, %rhs) : (!ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_subtract_i32
func.func @test_subtract_i32(%lhs: !ttype_i32, %rhs: !ttype_i32) -> (!ttype_i32) {
  // CHECK: ttkernel.sub_int_tile_init
  // CHECK: ttkernel.sub_int_tile({{.*}}, <si32>)
  %0 = "ttir.subtract"(%lhs, %rhs) : (!ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_multiply_i32
func.func @test_multiply_i32(%lhs: !ttype_i32, %rhs: !ttype_i32) -> (!ttype_i32) {
  // CHECK: ttkernel.mul_int_tile_init(<si32>)
  // CHECK: ttkernel.mul_int_tile({{.*}}, <si32>)
  %0 = "ttir.multiply"(%lhs, %rhs) : (!ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_maximum_i32
func.func @test_maximum_i32(%lhs: !ttype_i32, %rhs: !ttype_i32) -> (!ttype_i32) {
  // CHECK: ttkernel.binary_max_int32_tile_init
  // CHECK: ttkernel.binary_max_int32_tile(
  %0 = "ttir.maximum"(%lhs, %rhs) : (!ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_minimum_i32
func.func @test_minimum_i32(%lhs: !ttype_i32, %rhs: !ttype_i32) -> (!ttype_i32) {
  // CHECK: ttkernel.binary_min_int32_tile_init
  // CHECK: ttkernel.binary_min_int32_tile(
  %0 = "ttir.minimum"(%lhs, %rhs) : (!ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}
