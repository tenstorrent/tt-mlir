// RUN: ttmlir-opt --ttir-fusion %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<1x32x128x128xf32>, %arg1: tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32> {
    // CHECK-NOT: "ttir.broadcast"
    %dps0 = tensor.empty() : tensor<1x32x128x128xf32>
    %0 = "ttir.broadcast"(%arg1, %dps0) {dimension = [3 : i64]} : (tensor<1x32x128x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32>
    %dps1 = tensor.empty() : tensor<1x32x128x128xf32>
    %1 = "ttir.add"(%arg0, %0, %dps1) {operandSegmentSizes = array<i32: 2, 1>} : (tensor<1x32x128x128xf32>, tensor<1x32x128x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32>
    return %1 : tensor<1x32x128x128xf32>
  }
}
