// RUN: ttmlir-opt --ttir-fusion %s | FileCheck %s
module attributes {} {
  func.func @keep_dim_3(%arg0: tensor<1x32x128x128xf32>) -> tensor<1x1x128x1xf32> {
    // CHECK-NOT:  "ttir.reshape"
    %dps1 = tensor.empty() : tensor<1x128xf32>
    %1 = "ttir.sum"(%arg0, %dps1) {keep_dim = false, dim_arg = [3 : i32, 1 : i32]} : (tensor<1x32x128x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
    %dps2 = tensor.empty() : tensor<1x1x128x1xf32>
    %2 = "ttir.reshape"(%1, %dps2) <{shape = [1: i32, 1: i32, 128: i32, 1: i32]}> : (tensor<1x128xf32>, tensor<1x1x128x1xf32>) -> tensor<1x1x128x1xf32>
    return %2 : tensor<1x1x128x1xf32>
  }
}


module attributes {} {
  func.func @lose_dim_3(%arg0: tensor<1x32x128x128xf32>) -> tensor<1x128xf32> {
    // CHECK-NOT:  "ttir.reshape"
    %dps1 = tensor.empty() : tensor<1x1x128x1xf32>
    %1 = "ttir.sum"(%arg0, %dps1) {keep_dim = true, dim_arg = [1 : i32, 3 : i32]} : (tensor<1x32x128x128xf32>, tensor<1x1x128x1xf32>) -> tensor<1x1x128x1xf32>
    %dps2 = tensor.empty() : tensor<1x128xf32>
    %2 = "ttir.reshape"(%1, %dps2) <{shape = [1: i32, 128: i32]}> : (tensor<1x1x128x1xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
    return %2 : tensor<1x128xf32>
  }
}
