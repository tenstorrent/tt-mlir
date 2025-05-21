// RUN: not ttmlir-opt %s 2>&1 | FileCheck %s

// CHECK: error: 'ttir.full' op expected shape (2, 3), got (3, 2)
func.func @full_shape_mismatch() -> tensor<2x3xi32> {
  %0 = "ttir.full"() <{shape = array<i32: 3, 2>, fill_value = 42 : i32}> : () -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}
