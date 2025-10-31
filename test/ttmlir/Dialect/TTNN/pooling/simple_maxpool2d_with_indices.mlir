// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @forward(%arg0: tensor<1x128x128x32xf32>) -> (tensor<1x64x64x32xf32>, tensor<1x64x64x32xi32>) {
    %0 = ttir.empty() : tensor<1x64x64x32xf32>
    %1 = ttir.empty() : tensor<1x64x64x32xi32>
    // CHECK: = "ttnn.max_pool2d_with_indices"
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{kernel = array<i32: 2, 2>, stride = array<i32: 2, 2>, dilation = array<i32: 1, 1>, padding = array<i32: 0, 0, 0, 0>, ceil_mode = false}> : (tensor<1x128x128x32xf32>, tensor<1x64x64x32xf32>, tensor<1x64x64x32xi32>) -> (tensor<1x64x64x32xf32>, tensor<1x64x64x32xi32>)
    return %2, %3 : tensor<1x64x64x32xf32>, tensor<1x64x64x32xi32>
  }
}
