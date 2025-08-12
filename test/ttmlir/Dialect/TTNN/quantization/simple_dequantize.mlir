// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-const-eval=false" -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @dequantize_per_tensor_scale_per_tensor_zp(%arg0: tensor<1x3x224x224x!quant.uniform<i32:f32, 2.000000e-02>>) -> tensor<1x3x224x224xf32> {
    // CHECK-LABEL: func.func @dequantize_per_tensor_scale_per_tensor_zp(
    %0 = ttir.empty() : tensor<1x3x224x224xf32>
    // CHECK: "ttnn.get_device"
    // CHECK: "ttnn.full"
    // CHECK-SAME: fill_value = 0
    // CHECK: "ttnn.full"
    // CHECK-SAME: fill_value = 2.000000e-02
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
  func.func @dequantize_per_axis_scale_per_tensor_zp(%arg0: tensor<3x3x7x7x!quant.uniform<i32:f32:0, {2.000000e-02,1.000000e-02,5.000000e-03}>>) -> tensor<3x3x7x7xf32> {
    // CHECK-LABEL: func.func @dequantize_per_axis_scale_per_tensor_zp(
    %0 = ttir.empty() : tensor<3x3x7x7xf32>
    // CHECK: "ttnn.get_device"
    // CHECK: "ttnn.full"
    // CHECK-SAME: fill_value = 0
    // CHECK-SAME: -> tensor<3xsi32,
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense<[2.000000e-02, 0.00999999977, 5.000000e-03]> : tensor<3xf32>
    // CHECK-SAME: -> tensor<3xf32,
    // CHECK: "ttnn.dequantize"
    // CHECK-SAME: axis = 0 : i32
    // CHECK-SAME: output_dtype = #ttcore.supportedDataTypes<f32>
    // CHECK-SAME: tensor<3x3x7x7x!quant.uniform<i32:f32:0, {2.000000e-02,1.000000e-02,5.000000e-03}>,
    // CHECK-SAME: -> tensor<3x3x7x7xf32,
    %1 = "ttir.constant"() <{value = dense<[2.000000e-02, 0.00999999977, 5.000000e-03]> : tensor<3xf32>}> : () -> tensor<3xf32>
    %2 = "ttir.constant"() <{value = dense<0> : tensor<3xi32>}> : () -> tensor<3xi32>
    %3 = ttir.empty() : tensor<3x3x7x7xf32>
    %4 = "ttir.dequantize_unrolled"(%arg0, %1, %2, %3) <{axis = 0 : i32}> : (tensor<3x3x7x7x!quant.uniform<i32:f32:0, {2.000000e-02,1.000000e-02,5.000000e-03}>>, tensor<3xf32>, tensor<3xi32>, tensor<3x3x7x7xf32>) -> tensor<3x3x7x7xf32>
    return %4 : tensor<3x3x7x7xf32>
  }
  func.func @dequantize_per_axis_scale_per_axis_zp(%arg0: tensor<3x3x7x7x!quant.uniform<i32:f32:0, {2.000000e-02:10,1.000000e-02:20,5.000000e-03:30}>>) -> tensor<3x3x7x7xf32> {
    // CHECK-LABEL: func.func @dequantize_per_axis_scale_per_axis_zp(
    %0 = ttir.empty() : tensor<3x3x7x7xf32>
    // CHECK: "ttnn.get_device"
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense<[10, 20, 30]> : tensor<3xsi32>
    // CHECK-SAME: -> tensor<3xsi32,
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense<[2.000000e-02, 0.00999999977, 5.000000e-03]> : tensor<3xf32>
    // CHECK-SAME: -> tensor<3xf32,
    // CHECK: "ttnn.dequantize"
    // CHECK-SAME: axis = 0 : i32
    // CHECK-SAME: output_dtype = #ttcore.supportedDataTypes<f32>
    // CHECK-SAME: tensor<3x3x7x7x!quant.uniform<i32:f32:0, {2.000000e-02:10,1.000000e-02:20,5.000000e-03:30}>,
    // CHECK-SAME: -> tensor<3x3x7x7xf32,
    %1 = "ttir.constant"() <{value = dense<[2.000000e-02, 0.00999999977, 5.000000e-03]> : tensor<3xf32>}> : () -> tensor<3xf32>
    %2 = "ttir.constant"() <{value = dense<[10, 20, 30]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %3 = ttir.empty() : tensor<3x3x7x7xf32>
    %4 = "ttir.dequantize_unrolled"(%arg0, %1, %2, %3) <{axis = 0 : i32}> : (tensor<3x3x7x7x!quant.uniform<i32:f32:0, {2.000000e-02:10,1.000000e-02:20,5.000000e-03:30}>>, tensor<3xf32>, tensor<3xi32>, tensor<3x3x7x7xf32>) -> tensor<3x3x7x7xf32>
    return %4 : tensor<3x3x7x7xf32>
  }
}
