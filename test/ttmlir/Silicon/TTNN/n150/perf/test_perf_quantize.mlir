// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-const-eval=false" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module {
  func.func @quantize_per_tensor_scale_per_tensor_zp(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>> {
    // CHECK-LABEL: func.func @quantize_per_tensor_scale_per_tensor_zp(
    %0 = ttir.empty() : tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>
    // CHECK: "ttnn.full"
    // CHECK-SAME: fill_value = 0
    // CHECK-SAME: -> tensor<1xui32,
    // CHECK: "ttnn.full"
    // CHECK-SAME: fill_value = 2.000000e-02 : f32
    // CHECK-SAME: -> tensor<1xf32,
    // CHECK: "ttnn.quantize"
    // CHECK-SAME: {output_dtype = #tt.supportedDataTypes<si32>}
    // CHECK-SAME: tensor<1x3x224x224xf32
    // CHECK-SAME: -> tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>,
    %1 = "ttir.constant"() <{value = dense<2.000000e-02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = ttir.empty() : tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>
    %4 = "ttir.quantize_unrolled"(%arg0, %1, %2, %3) : (tensor<1x3x224x224xf32>, tensor<1xf32>, tensor<1xi32>, tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>) -> tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>
    return %4 : tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>
  }
}
