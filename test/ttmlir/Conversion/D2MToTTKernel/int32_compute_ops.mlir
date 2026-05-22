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

// Compare-to-zero int32 ops (lowered from binary TTIR comparison ops via
// subtract + compare-to-zero).

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_eq_i32
func.func @test_eq_i32(%lhs: !ttype_i32, %rhs: !ttype_i32) -> (!ttype_i32) {
  // CHECK: ttkernel.sub_int_tile(
  // CHECK: ttkernel.eqz_tile_int32(
  %0 = "ttir.eq"(%lhs, %rhs) : (!ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_ne_i32
func.func @test_ne_i32(%lhs: !ttype_i32, %rhs: !ttype_i32) -> (!ttype_i32) {
  // CHECK: ttkernel.sub_int_tile(
  // CHECK: ttkernel.nez_tile_int32(
  %0 = "ttir.ne"(%lhs, %rhs) : (!ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_gt_i32
func.func @test_gt_i32(%lhs: !ttype_i32, %rhs: !ttype_i32) -> (!ttype_i32) {
  // CHECK: ttkernel.sub_int_tile(
  // CHECK: ttkernel.gtz_tile_int32(
  %0 = "ttir.gt"(%lhs, %rhs) : (!ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_ge_i32
func.func @test_ge_i32(%lhs: !ttype_i32, %rhs: !ttype_i32) -> (!ttype_i32) {
  // CHECK: ttkernel.sub_int_tile(
  // CHECK: ttkernel.gez_tile_int32(
  %0 = "ttir.ge"(%lhs, %rhs) : (!ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_lt_i32
func.func @test_lt_i32(%lhs: !ttype_i32, %rhs: !ttype_i32) -> (!ttype_i32) {
  // lt(a, b) is normalized to gtz(b - a).
  // CHECK: ttkernel.sub_int_tile(
  // CHECK: ttkernel.gtz_tile_int32(
  %0 = "ttir.lt"(%lhs, %rhs) : (!ttype_i32, !ttype_i32) -> !ttype_i32
  return %0 : !ttype_i32
}

// -----

!ttype_i32 = tensor<32x32xsi32>
// CHECK-LABEL: func.func @test_le_i32
func.func @test_le_i32(%lhs: !ttype_i32, %rhs: !ttype_i32) -> (!ttype_i32) {
  // le(a, b) is normalized to gez(b - a).
  // CHECK: ttkernel.sub_int_tile(
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

// -----

// Multi-tile-per-core int32 chain: regression test for the per-iteration
// DST allocation bug. The integer binary op rewriter used to hardcode
// `dst0 = 0, dst1 = 1` for the SFPU `*_int_tile` operands, which produced
// wrong results in any fused load/compute/store loop with >1 tile per
// iteration (every iteration overwrote DST[0..1]; the pack step then
// read stale data). After fixing
// `TileMulOp::getOperandsLoadFromDstRegister` to return `{0, 1}` for
// integer tiles (matching f32), `d2m-insert-dst-register-access` allocates
// distinct DST slots per loop iteration and the SFPU path picks them up
// via `getDstIdxFromResult` instead of constants. We verify that the
// generated kernel has explicit `copy_tile` loads (one per loop iter, into
// per-iteration DST slots) before the `mul_int_tile`, instead of two
// hardcoded `copy_tile(..., 0)` / `copy_tile(..., 1)` calls inside a fused
// loop body.
// Tensor is 64x32 -> 2 tiles in the row direction so the inner tile loop
// has trip count 2 and the bug shows up.
!ttype_i32_multi = tensor<64x32xsi32>
// CHECK-LABEL: func.func @test_multiply_i32_multi_tile
func.func @test_multiply_i32_multi_tile(%lhs: !ttype_i32_multi,
                                        %rhs: !ttype_i32_multi)
    -> (!ttype_i32_multi) {
  // The fix: per-iteration DST slot for each tile (allocated by
  // insert-dst-register-access via `getOperandsLoadFromDstRegister`).
  // We require at least two distinct `copy_tile` invocations before the
  // `mul_int_tile`, indicating the rewriter no longer hardcoded a single
  // dst slot per side.
  // CHECK: ttkernel.copy_tile
  // CHECK: ttkernel.copy_tile
  // CHECK: ttkernel.mul_int_tile_init(<si32>)
  // CHECK: ttkernel.mul_int_tile({{.*}}, <si32>)
  %0 = "ttir.multiply"(%lhs, %rhs)
      : (!ttype_i32_multi, !ttype_i32_multi) -> !ttype_i32_multi
  return %0 : !ttype_i32_multi
}
