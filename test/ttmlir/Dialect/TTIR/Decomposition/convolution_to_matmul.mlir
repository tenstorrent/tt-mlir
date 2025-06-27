// RUN: ttmlir-opt --ttir-conv2d-to-matmul %s | FileCheck %s

module {
  func.func @test_conv_1x1_to_matmul(%arg0: tensor<1x512x7x7xf32>, %arg1: tensor<2048x512x1x1xf32>) -> tensor<1x2048x7x7xf32> {
    // CHECK-LABEL: func.func @test_conv_1x1_to_matmul
    // CHECK: %[[EMPTY0:[0-9]+]] = ttir.empty() : tensor<1x2048x7x7xf32>
    // CHECK: %[[EMPTY1:[0-9]+]] = ttir.empty() : tensor<1x7x7x512xf32>
    // CHECK: %[[PERM0:[0-9]+]] = "ttir.permute"(%arg0, %[[EMPTY1]]) <{permutation = array<i64: 0, 2, 3, 1>}>
    // CHECK: %[[EMPTY2:[0-9]+]] = ttir.empty() : tensor<1x7x7x2048xf32>
    // CHECK: %[[EMPTY3:[0-9]+]] = ttir.empty() : tensor<49x512xf32>
    // CHECK: %[[RESHAPE0:[0-9]+]] = "ttir.reshape"(%[[PERM0]], %[[EMPTY3]]) <{shape = [49 : i32, 512 : i32]}>
    // CHECK: %[[EMPTY4:[0-9]+]] = ttir.empty() : tensor<512x2048x1x1xf32>
    // CHECK: %[[PERM1:[0-9]+]] = "ttir.permute"(%arg1, %[[EMPTY4]]) <{permutation = array<i64: 1, 0, 2, 3>}>
    // CHECK: %[[EMPTY5:[0-9]+]] = ttir.empty() : tensor<512x2048xf32>
    // CHECK: %[[RESHAPE1:[0-9]+]] = "ttir.reshape"(%[[PERM1]], %[[EMPTY5]]) <{shape = [512 : i32, 2048 : i32]}>
    // CHECK: %[[EMPTY6:[0-9]+]] = ttir.empty() : tensor<49x2048xf32>
    // CHECK: %[[MATMUL:[0-9]+]] = "ttir.matmul"(%[[RESHAPE0]], %[[RESHAPE1]], %[[EMPTY6]]) <{transpose_a = false, transpose_b = false}>
    // CHECK: %[[RESHAPE2:[0-9]+]] = "ttir.reshape"(%[[MATMUL]], %[[EMPTY2]]) <{shape = [1 : i32, 7 : i32, 7 : i32, 2048 : i32]}>
    // CHECK: %[[RESULT:[0-9]+]] = "ttir.permute"(%[[RESHAPE2]], %[[EMPTY0]]) <{permutation = array<i64: 0, 3, 1, 2>}>
    // CHECK: return %[[RESULT]]
    %0 = ttir.empty() : tensor<1x2048x7x7xf32>
    %1 = ttir.empty() : tensor<1x7x7x512xf32>
    %2 = "ttir.permute"(%arg0, %1) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x512x7x7xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %3 = ttir.empty() : tensor<2048x512x1x1xf32>
    %4 = "ttir.permute"(%arg1, %3) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<2048x512x1x1xf32>, tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %5 = ttir.empty() : tensor<1x7x7x2048xf32>
    %6 = "ttir.conv2d"(%2, %4, %5) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> : (tensor<1x7x7x512xf32>, tensor<2048x512x1x1xf32>, tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    %7 = "ttir.permute"(%6, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x7x7x2048xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    return %7 : tensor<1x2048x7x7xf32>
  }
}
