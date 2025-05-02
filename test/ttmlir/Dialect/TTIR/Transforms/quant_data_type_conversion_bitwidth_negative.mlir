// RUN: not ttmlir-opt --ttir-quant-data-type-conversion="target-bit-width=12" %s 2>&1 | FileCheck %s

func.func @requantize_datatype_bad_bit_width_negative(%arg0: tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 2.000000e-01>> {
  // CHECK: error: Invalid quantization bit width (must be 8, 16, 32, or 64).
  %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32, 2.000000e-01>>
  %1 = "ttir.requantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>, tensor<1x3x320x320x!quant.uniform<i32:f32, 2.000000e-01>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 2.000000e-01>>
  return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32, 2.000000e-01>>
}
