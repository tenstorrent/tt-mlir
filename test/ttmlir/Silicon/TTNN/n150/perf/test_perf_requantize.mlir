// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module {
  func.func @requantize_test(%arg0: tensor<1x3x320x320x!quant.uniform<i32:f32, 0.1>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>> {
    // CHECK-LABEL: func.func @requantize_test(
    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
    %1 = "ttir.requantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i32:f32, 0.1>>, tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
    // CHECK: "ttnn.requantize"(%arg0)
    // CHECK-SAME: {in_scale = 1.000000e-01 : f32, in_zero_point = 0 : i32, out_scale = 2.000000e-01 : f32, out_zero_point = 0 : i32, output_dtype = #tt.supportedDataTypes<si32>}
    // CHECK-SAME: tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>,
    // CHECK-SAME: -> tensor<1x3x320x320x!quant.uniform<i32:f32, 2.000000e-01>,
    return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
  }
}
