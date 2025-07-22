// RUN: ttmlir-opt --ttir-erase-inverse-ops -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @main(%arg0: tensor<64x3x7x7xf32>, %arg1: tensor<1x3x256x256xf32>, %arg2: tensor<1x64x128x128xf32>, %arg3: tensor<512x64x7x7xf32>) -> tensor<1x512x64x64xf32> {
    // This module tests whether the permute on the output of the first conv2d is erased along with the permute on the input of the second conv2d.
    // CHECK: %[[CONV0:[0-9]+]] = "ttir.conv2d"
    // CHECK: %[[ADD:[0-9]+]] = "ttir.add"(%{{.+}}, %[[CONV0]],
    // CHECK: %[[EXP:[0-9]+]] = "ttir.exp"(%[[ADD]],
    // CHECK: %[[CONV1:[0-9]+]] = "ttir.conv2d"(%[[EXP]],
    %0 = tensor.empty() : tensor<1x64x128x128xf32>
    %1 = tensor.empty() : tensor<1x256x256x3xf32>
    %2 = "ttir.permute"(%arg1, %1) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x3x256x256xf32>, tensor<1x256x256x3xf32>) -> tensor<1x256x256x3xf32>
    %3 = tensor.empty() : tensor<64x3x7x7xf32>
    %4 = "ttir.permute"(%arg0, %3) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<64x3x7x7xf32>, tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf32>
    %5 = tensor.empty() : tensor<1x128x128x64xf32>
    %6 = "ttir.conv2d"(%2, %4, %5) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 3, 3, 3, 3>, stride = array<i32: 2, 2>}> : (tensor<1x256x256x3xf32>, tensor<64x3x7x7xf32>, tensor<1x128x128x64xf32>) -> tensor<1x128x128x64xf32>
    %7 = "ttir.permute"(%6, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x128x128x64xf32>, tensor<1x64x128x128xf32>) -> tensor<1x64x128x128xf32>
    %8 = tensor.empty() : tensor<1x64x128x128xf32>
    %9 = "ttir.add"(%arg2, %7, %8) : (tensor<1x64x128x128xf32>, tensor<1x64x128x128xf32>, tensor<1x64x128x128xf32>) -> tensor<1x64x128x128xf32>
    %10 = tensor.empty() : tensor<1x64x128x128xf32>
    %11 = "ttir.exp"(%9, %10) : (tensor<1x64x128x128xf32>, tensor<1x64x128x128xf32>) -> tensor<1x64x128x128xf32>
    %12 = tensor.empty() : tensor<1x512x64x64xf32>
    %13 = tensor.empty() : tensor<1x128x128x64xf32>
    %14 = "ttir.permute"(%11, %13) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x64x128x128xf32>, tensor<1x128x128x64xf32>) -> tensor<1x128x128x64xf32>
    %15 = tensor.empty() : tensor<512x64x7x7xf32>
    %16 = "ttir.permute"(%arg3, %15) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<512x64x7x7xf32>, tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xf32>
    %17 = tensor.empty() : tensor<1x64x64x512xf32>
    %18 = "ttir.conv2d"(%14, %16, %17) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 3, 3, 3, 3>, stride = array<i32: 2, 2>}> : (tensor<1x128x128x64xf32>, tensor<512x64x7x7xf32>, tensor<1x64x64x512xf32>) -> tensor<1x64x64x512xf32>
    %19 = "ttir.permute"(%18, %12) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x64x64x512xf32>, tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32>
    return %19 : tensor<1x512x64x64xf32>
  }
}
