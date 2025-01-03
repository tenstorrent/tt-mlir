// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module {
  func.func @main(%arg0: tensor<1x16x32xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
    // CHECK: %{{[0-9]+}} = "ttnn.repeat"
    // CHECK-SAME: shape = [1 : i32, 16 : i32, 1 : i32]
    %0 = tensor.empty() : tensor<1x16x32xf32>
    %1 = "ttir.repeat"(%arg1, %0) <{repeat_dimensions = [1 : i32, 16 : i32, 1 : i32]}> : (tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %2 = tensor.empty() : tensor<1x16x32xf32>
    %3 = "ttir.multiply"(%arg0, %1, %2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x16x32xf32>, tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    return %3 : tensor<1x16x32xf32>
  }
}

module {
  func.func public @main(%arg0: tensor<1xf32>, %arg1: tensor<512x512xf32>) -> (tensor<512x512xf32>) {
    // CHECK: %{{[0-9]+}} = "ttnn.reshape"
    // CHECK: %{{[0-9]+}} = "ttnn.repeat"
    // CHECK-SAME: shape = [512 : i32, 512 : i32]
    %0 = tensor.empty() : tensor<1x1xf32>
    %1 = "ttir.reshape"(%arg0, %0) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = tensor.empty() : tensor<512x512xf32>
    %3 = "ttir.repeat"(%1, %2) <{repeat_dimensions = [512 : i32, 512 : i32]}> : (tensor<1x1xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
    %4 = tensor.empty() : tensor<512x512xf32>
    %5 = "ttir.maximum"(%3, %arg1, %4) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<512x512xf32>, tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
    return %5 : tensor<512x512xf32>
  }
}
