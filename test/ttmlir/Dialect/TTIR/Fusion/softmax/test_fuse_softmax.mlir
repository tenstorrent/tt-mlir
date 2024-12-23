// RUN: ttmlir-opt --ttir-fusion %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32> {
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
