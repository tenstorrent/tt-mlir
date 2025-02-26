// RUN: ttmlir-opt --remove-dead-values %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = tensor.empty() : tensor<64x128xf32>
    // CHECK: = "ttir.multiply"
    %1 = "ttir.multiply"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %2 = tensor.empty() : tensor<64x128xf32>
    // CHECK-NOT: = "ttir.add"
    %3 = "ttir.add"(%arg0, %arg1, %2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %4 = tensor.empty() : tensor<64x128xf32>
    // CHECK-NOT: = "ttir.subtract"
    %5 = "ttir.subtract"(%arg0, %arg1, %4) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %6 = tensor.empty() : tensor<64x128xf32>
    // CHECK-NOT: = "ttir.div"
    %7 = "ttir.div"(%arg0, %arg1, %6) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    %8 = tensor.empty() : tensor<64x128xf32>
    // CHECK-NOT: = "ttir.eq"
    %9 = "ttir.eq"(%arg0, %arg1, %8) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}
