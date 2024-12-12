// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xf32>) -> tensor<64x128xbf16> {
    %0 = tensor.empty() : tensor<64x128xbf16>
    %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xf32>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // CHECK: %[[C:.*]] = "ttnn.typecast"
    // CHECK-SAME: tensor<64x128xf32,
    // CHECK-SAME: tensor<64x128xbf16,
    return %1 : tensor<64x128xbf16>
  }

  func.func @cast_fold(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CEHCK-LABEL: func.func @cast_fold
    %0 = tensor.empty() : tensor<64x128xf32>
    %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK-NOT: typecast
    // CHECK: return %arg0 : tensor<64x128xf32
    return %1 : tensor<64x128xf32>
  }

  func.func @cast_fold_add(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK-LABEL: func.func @cast_fold_add
    // CHECK-NOT: typecast
    // CHECK: ttnn.add
    %0 = tensor.empty() : tensor<64x128xf32>
    %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %2 = tensor.empty() : tensor<64x128xf32>
    %3 = "ttir.add"(%1, %arg1, %2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %3 : tensor<64x128xf32>
  }
}
