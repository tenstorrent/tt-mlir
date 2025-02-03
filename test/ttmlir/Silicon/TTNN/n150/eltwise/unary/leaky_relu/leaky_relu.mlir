// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

func.func @leaky_relu(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK: %[[C:.*]] = "ttnn.empty"
    %0 = tensor.empty() : tensor<64x128xf32>
    // CHECK: %[[C:.*]] = "ttnn.leaky_relu"
    %1 = "ttir.leaky_relu"(%arg0, %0) <{parameter = 0.01 : f32, operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
}
