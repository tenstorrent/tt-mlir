// RUN: not ttmlir-opt --ttir-quant-data-type-conversion="quant-data-type=int8" %s 2>&1 | FileCheck %s --check-prefix=CHECK-BAD-INT32
// RUN: not ttmlir-opt --ttir-quant-data-type-conversion="quant-data-type=float32" %s 2>&1 | FileCheck %s --check-prefix=CHECK-BAD-FLOAT32
// XFAIL: *
func.func @requantize_datatype_negative(%arg0: tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 2.000000e-01>> {
  // CHECK-BAD-INT32: Assertion `intType.getWidth() >= quantType.getStorageType().getIntOrFloatBitWidth() && "Target integer type is smaller than quantized type. Out of range."' failed.
  // CHECK-BAD-FLOAT32: Assertion `targetIntType && ("Invalid target bit width: " + targetBitWidth).c_str()' failed.
  %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32, 2.000000e-01>>
  %1 = "ttir.requantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>>, tensor<1x3x320x320x!quant.uniform<i32:f32, 2.000000e-01>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 2.000000e-01>>
  return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32, 2.000000e-01>>
}
