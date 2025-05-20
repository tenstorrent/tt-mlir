// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for TTIR FullOp

// CHECK: error: 'ttir.full' op expected shape (2, 3), got (3, 2)
func.func @full_shape_mismatch() -> tensor<2x3xi32> {
  %0 = "ttir.full"() <{shape = array<i32: 3, 2>, fill_value = 42 : i32}> : () -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// -----
// CHECK: error: 'ttir.full' op expected fill value of floating point type, got integral type
func.func @full_type_mismatch_float_to_int() -> tensor<2x3xi32> {
  %0 = "ttir.full"() <{shape = array<i32: 2, 3>, fill_value = 1.0 : f32}> : () -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// -----
// CHECK: error: 'ttir.full' op expected fill value of integral type, got floating point type
func.func @full_type_mismatch_int_to_float() -> tensor<2x3xf32> {
  %0 = "ttir.full"() <{shape = array<i32: 2, 3>, fill_value = 42 : i32}> : () -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
