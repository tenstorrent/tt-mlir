// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

func.func @div(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  %1 = "ttir.div"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  // CHECK: "ttnn.divide"
  // CHECK-SAME: tensor<64x128xf32
  // CHECK-SAME: tensor<64x128xf32
  // CHECK-SAME: -> tensor<64x128xf32
  return %1 : tensor<64x128xf32>
}

func.func @div_broadcast(%arg0: tensor<1x1xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK-LABEL: @div_broadcast
  %0 = ttir.empty() : tensor<64x128xf32>
  %1 = "ttir.broadcast"(%arg0, %0) <{broadcast_dimensions = array<i64: 64, 128>}> : (tensor<1x1xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  %2 = ttir.empty() : tensor<64x128xf32>
  // CHECK: %{{[0-9]+}} = "ttnn.divide"
  // CHECK-SAME: tensor<64x128xf32,
  // CHECK-SAME: tensor<64x128xf32,
  // CHECK-SAME: -> tensor<64x128xf32,
  %3 = "ttir.div"(%1, %arg1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %3 : tensor<64x128xf32>
}
