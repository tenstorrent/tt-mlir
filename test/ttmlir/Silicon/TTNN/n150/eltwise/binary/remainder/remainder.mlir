// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

func.func @remainder(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = ttir.empty() : tensor<32x32xf32>
  %1 = "ttir.remainder"(%arg0, %arg1, %0) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: "ttnn.remainder"
  // CHECK-SAME: tensor<32x32xf32
  // CHECK-SAME: tensor<32x32xf32
  // CHECK-SAME: -> tensor<32x32xf32
  return %1 : tensor<32x32xf32>
}

func.func @remainder_broadcast(%arg0: tensor<64x128xf32>, %arg1: tensor<1x1xf32>) -> tensor<64x128xf32> {
  // CHECK-LABEL: @remainder_broadcast
  %0 = ttir.empty() : tensor<64x128xf32>
  %1 = "ttir.broadcast"(%arg1, %0) <{broadcast_dimensions = array<i64: 64, 128>}> : (tensor<1x1xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  %2 = ttir.empty() : tensor<64x128xf32>
  // CHECK: %{{[0-9]+}} = "ttnn.remainder"
  // CHECK-SAME: tensor<64x128xf32,
  // CHECK-SAME: tensor<1x1xf32,
  // CHECK-SAME: -> tensor<64x128xf32,
  %3 = "ttir.remainder"(%arg0, %1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %3 : tensor<64x128xf32>
}
