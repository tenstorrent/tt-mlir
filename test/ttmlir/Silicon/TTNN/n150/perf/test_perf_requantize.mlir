// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-const-eval=false" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module {
func.func @requantize_per_tensor_scales_per_tensor_zps(%arg0: tensor<1x3x320x320x!quant.uniform<i32:f32, 0.1>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>> {
    // CHECK-LABEL: func.func @requantize_per_tensor_scales_per_tensor_zps(
    %0 = ttir.empty() : tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
    // CHECK: "ttnn.full"
    // CHECK-SAME: fill_value = 2.000000e-01 : f32
    // CHECK-SAME: -> tensor<1xf32,
    // CHECK: "ttnn.full"
    // CHECK-SAME: fill_value = 0 : i32
    // CHECK-SAME: -> tensor<1xi32,
    // CHECK: "ttnn.full"
    // CHECK-SAME: fill_value = 1.000000e-01 : f32
    // CHECK-SAME: -> tensor<1xf32,
    // CHECK: "ttnn.requantize"
    // CHECK-SAME: {output_dtype = #ttcore.supportedDataTypes<si32>}
    // CHECK-SAME: tensor<1x3x320x320x!quant.uniform<i32:f32, 1.000000e-01>,
    // CHECK-SAME: -> tensor<1x3x320x320x!quant.uniform<i32:f32, 2.000000e-01>,
    %1 = "ttir.requantize"(%arg0, %0) : (tensor<1x3x320x320x!quant.uniform<i32:f32, 0.1>>, tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>) -> tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
    return %1 : tensor<1x3x320x320x!quant.uniform<i32:f32, 0.2>>
  }
}
