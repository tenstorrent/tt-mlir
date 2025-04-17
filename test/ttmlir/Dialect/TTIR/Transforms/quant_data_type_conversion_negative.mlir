// RUN: not ttmlir-opt --ttir-quant-data-type-conversion="quant-bit-width=8" %s 2>&1 | FileCheck %s
// XFAIL: *
func.func @requantize_datatype_negative(%arg0: tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 2.000000e-01>> {
  // CHECK: Assertion `intType.getWidth() >= quantType.getStorageType().getIntOrFloatBitWidth() && "Target integer type is smaller than quantized type. Out of range."' failed.
  %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32, 2.000000e-01>>
  %1 = "ttir.requantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>, tensor<1x3x320x320x!quant.uniform<i32:f32, 2.000000e-01>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 2.000000e-01>>
  return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32, 2.000000e-01>>
}
