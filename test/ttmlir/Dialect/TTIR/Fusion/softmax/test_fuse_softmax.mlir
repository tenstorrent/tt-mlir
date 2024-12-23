// RUN: ttmlir-opt --ttir-fusion %s | FileCheck %s
module attributes {} {
  func.func @softmax_pattern_with_explicit_broadcast(%arg0: tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32> {
    // CHECK: %[[C:.*]] = "ttir.softmax"[[C:.*]]
    %dps1 = tensor.empty() : tensor<1x32x128x128xf32>
    %1 = "ttir.exp"(%arg0, %dps1) {operandSegmentSizes = array<i32: 1, 1>} : (tensor<1x32x128x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32>
    %dps2 = tensor.empty() : tensor<1x32x128x1xf32>
    %2 = "ttir.sum"(%1, %dps2) {keep_dim = true, dim_arg = [3 : i32]} : (tensor<1x32x128x128xf32>, tensor<1x32x128x1xf32>) -> tensor<1x32x128x1xf32>
    %dps3 = tensor.empty() : tensor<1x32x128x128xf32>
    %3 = "ttir.broadcast"(%2, %dps3) {dimension = [3 : i64]} : (tensor<1x32x128x1xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32>
    %dps4 = tensor.empty() : tensor<1x32x128x128xf32>
    %4 = "ttir.div"(%1, %3, %dps4) {operandSegmentSizes = array<i32: 2, 1>} : (tensor<1x32x128x128xf32>, tensor<1x32x128x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32>
    return %4 : tensor<1x32x128x128xf32>
  }
}

module attributes {} {
  func.func @softmax_pattern_with_implicit_broadcast(%arg0: tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32> {
    // CHECK: %[[C:.*]] = "ttir.softmax"[[C:.*]]
    %dps1 = tensor.empty() : tensor<1x32x128x128xf32>
    %1 = "ttir.exp"(%arg0, %dps1) {operandSegmentSizes = array<i32: 1, 1>} : (tensor<1x32x128x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32>
    %dps2 = tensor.empty() : tensor<1x32x128x1xf32>
    %2 = "ttir.sum"(%1, %dps2) {keep_dim = true, dim_arg = [3 : i32]} : (tensor<1x32x128x128xf32>, tensor<1x32x128x1xf32>) -> tensor<1x32x128x1xf32>
    %dps3 = tensor.empty() : tensor<1x32x128x128xf32>
    %3 = "ttir.div"(%1, %2, %dps3) {operandSegmentSizes = array<i32: 2, 1>} : (tensor<1x32x128x128xf32>, tensor<1x32x128x1xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32>
    return %3 : tensor<1x32x128x128xf32>
  }
}

module attributes {} {
  func.func @softmax_pattern_with_fusable_keepdim_reduce_and_broadcast(%arg0: tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32> {
    // CHECK: %[[C:.*]] = "ttir.softmax"[[C:.*]]
    %dps1 = tensor.empty() : tensor<1x32x128x128xf32>
    %1 = "ttir.exp"(%arg0, %dps1) {operandSegmentSizes = array<i32: 1, 1>} : (tensor<1x32x128x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32>
    %dps2 = tensor.empty() : tensor<1x32x128xf32>
    %2 = "ttir.sum"(%1, %dps2) {keep_dim = false, dim_arg = [3 : i32]} : (tensor<1x32x128x128xf32>, tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    %dps3 = tensor.empty() : tensor<1x32x128x1xf32>
    %3 = "ttir.reshape"(%2, %dps3) {shape = [1: i32, 32: i32, 128: i32, 1: i32]} : (tensor<1x32x128xf32>, tensor<1x32x128x1xf32>) -> tensor<1x32x128x1xf32>
    %dps4 = tensor.empty() : tensor<1x32x128x128xf32>
    %4 = "ttir.broadcast"(%3, %dps4) {dimension = [3 : i64]} : (tensor<1x32x128x1xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32>
    %dps5 = tensor.empty() : tensor<1x32x128x128xf32>
    %5 = "ttir.div"(%1, %4, %dps4) {operandSegmentSizes = array<i32: 2, 1>} : (tensor<1x32x128x128xf32>, tensor<1x32x128x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32>
    return %5 : tensor<1x32x128x128xf32>
  }
}
