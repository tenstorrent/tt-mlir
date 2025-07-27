// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-const-eval=false" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module {
  func.func @quantize_per_tensor_scale_per_tensor_zp(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>> {
    // CHECK-LABEL: func.func @quantize_per_tensor_scale_per_tensor_zp(
    %0 = ttir.empty() : tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>
    // CHECK: "ttnn.full"
    // CHECK-SAME: fill_value = 0
    // CHECK-SAME: -> tensor<1xf32,
    // CHECK: "ttnn.full"
    // CHECK-SAME: fill_value = 2.000000e-02 : f32
    // CHECK-SAME: -> tensor<1xf32,
    // CHECK: "ttnn.quantize"
    // CHECK-SAME: {output_dtype = #ttcore.supportedDataTypes<si32>}
    // CHECK-SAME: tensor<1x3x224x224xf32
    // CHECK-SAME: -> tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>,
    %1 = "ttir.constant"() <{value = dense<2.000000e-02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = ttir.empty() : tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>
    %4 = "ttir.quantize_unrolled"(%arg0, %1, %2, %3) : (tensor<1x3x224x224xf32>, tensor<1xf32>, tensor<1xi32>, tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>) -> tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>
    return %4 : tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>
  }
  func.func @quantize_per_axis_scale_per_tensor_zp(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02,2.000000e-02,3.000000e-02}>> {
    // CHECK-LABEL: func.func @quantize_per_axis_scale_per_tensor_zp(
    %0 = ttir.empty() : tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02,2.000000e-02,3.000000e-02}>>
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense<[0.00999999977, 2.000000e-02, 3.000000e-02]> : tensor<3xf32>
    // CHECK-SAME: -> tensor<3xf32,
    // CHECK: "ttnn.full"
    // CHECK-SAME: fill_value = 0
    // CHECK-SAME: -> tensor<3xf32,
    // CHECK: "ttnn.quantize"
    // CHECK-SAME: axis = 1 : i32
    // CHECK-SAME: output_dtype = #ttcore.supportedDataTypes<si32>
    // CHECK-SAME: tensor<1x3x224x224xf32
    // CHECK-SAME: -> tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02,2.000000e-02,3.000000e-02}>,
    %1 = "ttir.constant"() <{value = dense<[0.00999999977, 2.000000e-02, 3.000000e-02]> : tensor<3xf32>}> : () -> tensor<3xf32>
    %2 = "ttir.constant"() <{value = dense<0> : tensor<3xi32>}> : () -> tensor<3xi32>
    %3 = ttir.empty() : tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02,2.000000e-02,3.000000e-02}>>
    %4 = "ttir.quantize_unrolled"(%arg0, %1, %2, %3) <{axis = 1 : i32}> : (tensor<1x3x224x224xf32>, tensor<3xf32>, tensor<3xi32>, tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02,2.000000e-02,3.000000e-02}>>) -> tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02,2.000000e-02,3.000000e-02}>>
    return %4 : tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02,2.000000e-02,3.000000e-02}>>
  }
  func.func @quantize_per_axis_scale_per_axis_zp(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02:10,2.000000e-02:20,3.000000e-02:30}>> {
    // CHECK-LABEL: func.func @quantize_per_axis_scale_per_axis_zp(
    %0 = ttir.empty() : tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02:10,2.000000e-02:20,3.000000e-02:30}>>
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense<[10, 20, 30]> : tensor<3xsi32>
    // CHECK-SAME: -> tensor<3xsi32,
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense<[0.00999999977, 2.000000e-02, 3.000000e-02]> : tensor<3xf32>
    // CHECK-SAME: -> tensor<3xf32,
    // CHECK: "ttnn.quantize"
    // CHECK-SAME: axis = 1 : i32
    // CHECK-SAME: output_dtype = #ttcore.supportedDataTypes<si32>
    // CHECK-SAME: tensor<1x3x224x224xf32
    // CHECK-SAME: -> tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02:10,2.000000e-02:20,3.000000e-02:30}>,
    %1 = "ttir.constant"() <{value = dense<[0.00999999977, 2.000000e-02, 3.000000e-02]> : tensor<3xf32>}> : () -> tensor<3xf32>
    %2 = "ttir.constant"() <{value = dense<[10, 20, 30]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %3 = ttir.empty() : tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02:10,2.000000e-02:20,3.000000e-02:30}>>
    %4 = "ttir.quantize_unrolled"(%arg0, %1, %2, %3) <{axis = 1 : i32}> : (tensor<1x3x224x224xf32>, tensor<3xf32>, tensor<3xi32>, tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02:10,2.000000e-02:20,3.000000e-02:30}>>) -> tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02:10,2.000000e-02:20,3.000000e-02:30}>>
    return %4 : tensor<1x3x224x224x!quant.uniform<i32:f32:1, {1.000000e-02:10,2.000000e-02:20,3.000000e-02:30}>>
  }
}
