// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

func.func @pow(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<32x32xf32>}> : () -> tensor<32x32xf32>
    %1 = ttir.empty() : tensor<32x32xf32>
    // CHECK: "ttnn.pow_scalar"
    // CHECK-SAME: <{exponent = 2.000000e+00 : f32}>
    // CHECK-SAME: tensor<32x32xf32,
    // CHECK-SAME: -> tensor<32x32xf32,
    %2 = "ttir.pow_tensor"(%arg0, %0, %1) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %2 : tensor<32x32xf32>
}
