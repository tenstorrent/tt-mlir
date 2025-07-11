// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

func.func @add(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  // CHECK: "ttnn.add"
  // CHECK-SAME: tensor<64x128xf32
  // CHECK-SAME: tensor<64x128xf32
  // CHECK-SAME: -> tensor<64x128xf32
  return %1 : tensor<64x128xf32>
}

func.func @add_scalars(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = ttir.empty() : tensor<f32>
  %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: "ttnn.add"
  // CHECK-SAME: tensor<f32
  // CHECK-SAME: tensor<f32
  // CHECK-SAME: -> tensor<f32
  return %1 : tensor<f32>
}
