// RUN: ttmlir-opt --split-input-file --ttir-to-ttmetal-fe-pipeline --ttir-to-ttmetal-me-pipeline --convert-d2m-to-ttkernel %s | FileCheck %s

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

// Binary int32 comparison ops. These currently share the float SFPU
// `*_binary_tile` kernels with the bf16/f32 path: the new TileEq/Ne/Gt/Ge ops
// are not in IntComputeOpMap (mirroring TileDivOp / TilePowOp), so the
// rewriter falls through to the default SFPU op regardless of dtype. If a
// dedicated integer kernel is added to tt-metal, register it in
// IntComputeOpMap and update these CHECKs.

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_eq_i32
func.func @test_eq_i32(%lhs: !ttype_i32, %rhs: !ttype_i32) -> (!ttype_i32) {
  // CHECK: ttkernel.eq_binary_tile_init
  // CHECK: ttkernel.eq_binary_tile(
  %0 = "ttir.eq"(%lhs, %rhs) : (!ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_ne_i32
func.func @test_ne_i32(%lhs: !ttype_i32, %rhs: !ttype_i32) -> (!ttype_i32) {
  // CHECK: ttkernel.ne_binary_tile_init
  // CHECK: ttkernel.ne_binary_tile(
  %0 = "ttir.ne"(%lhs, %rhs) : (!ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_gt_i32
func.func @test_gt_i32(%lhs: !ttype_i32, %rhs: !ttype_i32) -> (!ttype_i32) {
  // CHECK: ttkernel.gt_binary_tile_init
  // CHECK: ttkernel.gt_binary_tile(
  %0 = "ttir.gt"(%lhs, %rhs) : (!ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_ge_i32
func.func @test_ge_i32(%lhs: !ttype_i32, %rhs: !ttype_i32) -> (!ttype_i32) {
  // CHECK: ttkernel.ge_binary_tile_init
  // CHECK: ttkernel.ge_binary_tile(
  %0 = "ttir.ge"(%lhs, %rhs) : (!ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_lt_i32
func.func @test_lt_i32(%lhs: !ttype_i32, %rhs: !ttype_i32) -> (!ttype_i32) {
  // ttir.lt(a, b) is canonicalized to ttir.gt(b, a), so this lowers to gt_binary_tile.
  // CHECK: ttkernel.gt_binary_tile_init
  // CHECK: ttkernel.gt_binary_tile(
  %0 = "ttir.lt"(%lhs, %rhs) : (!ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_le_i32
func.func @test_le_i32(%lhs: !ttype_i32, %rhs: !ttype_i32) -> (!ttype_i32) {
  // ttir.le(a, b) is canonicalized to ttir.ge(b, a), so this lowers to ge_binary_tile.
  // CHECK: ttkernel.ge_binary_tile_init
  // CHECK: ttkernel.ge_binary_tile(
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
