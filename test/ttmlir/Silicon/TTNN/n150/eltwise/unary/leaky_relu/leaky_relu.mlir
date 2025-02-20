// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

func.func @leaky_relu(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = tensor.empty() : tensor<64x128xf32>
    %1 = "ttir.leaky_relu"(%arg0, %0) <{parameter = 0.01 : f32, operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: %[[RETURN_VALUE:[0-9]+]] = "ttnn.leaky_relu"(%arg0)
    // CHECK-SAME: <{parameter = 0.00999999977 : f32}>
    // CHECK-SAME: (tensor<64x128xf32, {{.*}}>)
    // CHECK-SAME: -> tensor<64x128xf32, {{.*}}>
    return %1 : tensor<64x128xf32>
    // CHECK: return %[[RETURN_VALUE]] : tensor<64x128xf32, {{.*}}>
}
