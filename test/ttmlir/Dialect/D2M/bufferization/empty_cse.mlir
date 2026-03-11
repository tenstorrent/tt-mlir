// RUN: ttmlir-opt --cse %s | FileCheck %s

// CHECK-LABEL: func.func @two_empty_ops_not_merged
// CHECK: d2m.empty
// CHECK: d2m.empty
func.func @two_empty_ops_not_merged() -> (tensor<2x4x!ttcore.tile<32x32, f32>>, tensor<2x4x!ttcore.tile<32x32, f32>>) {
  %0 = d2m.empty() : tensor<2x4x!ttcore.tile<32x32, f32>>
  %1 = d2m.empty() : tensor<2x4x!ttcore.tile<32x32, f32>>
  return %0, %1 : tensor<2x4x!ttcore.tile<32x32, f32>>, tensor<2x4x!ttcore.tile<32x32, f32>>
}
