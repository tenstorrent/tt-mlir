// RUN: ttmlir-opt -o %t %s
// RUN: FileCheck %s --input-file=%t

// This test verifies you can use tile types w/ the arith dialect
// which requires its types to conform to the FloatTypeInterface

// CHECK-NOT: error: 'arith.negf' op operand #0 must be floating-point-like

func.func @float_interface(%arg0: !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32> {
  %1 = arith.negf %arg0 : !ttcore.tile<32x32, f32>
  return %1 : !ttcore.tile<32x32, f32>
}

func.func @float_interface_element_type(%arg0: tensor<2x4x!ttcore.tile<32x32, f32>>) -> tensor<2x4x!ttcore.tile<32x32, f32>> {
  %1 = arith.negf %arg0 : tensor<2x4x!ttcore.tile<32x32, f32>>
  return %1 : tensor<2x4x!ttcore.tile<32x32, f32>>
}
