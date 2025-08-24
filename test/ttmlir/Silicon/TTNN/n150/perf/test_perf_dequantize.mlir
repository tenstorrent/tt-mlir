// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-const-eval=false" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module {
  func.func @dequantize_per_tensor_scale_per_tensor_zp(%arg0: tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>) -> tensor<1x3x224x224xf32> {
    // CHECK-LABEL: func.func @dequantize_per_tensor_scale_per_tensor_zp(
    %0 = ttir.empty() : tensor<1x3x224x224xf32>
    // CHECK: "ttnn.get_device"
    // CHECK: "ttnn.full"
    // CHECK-SAME: fill_value = 0
    // CHECK-SAME: -> tensor<1xsi32,
    // CHECK: "ttnn.full"
    // CHECK-SAME: fill_value = 2.000000e-02
    // CHECK-SAME: -> tensor<1xf32,
    // CHECK: "ttnn.dequantize"
    // CHECK-SAME: output_dtype = #ttcore.supportedDataTypes<f32>
    // CHECK-SAME: tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>,
    // CHECK-SAME: -> tensor<1x3x224x224xf32,
    %1 = "ttir.constant"() <{value = dense<2.000000e-02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = ttir.empty() : tensor<1x3x224x224xf32>
    %4 = "ttir.dequantize_unrolled"(%arg0, %1, %2, %3) : (tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>, tensor<1xf32>, tensor<1xi32>, tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32>
    return %4 : tensor<1x3x224x224xf32>
  }
}
