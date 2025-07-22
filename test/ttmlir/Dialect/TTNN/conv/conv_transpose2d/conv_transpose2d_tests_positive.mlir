// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @forward(%arg0: tensor<20x50x100x16xf32>, %arg1: tensor<16x32x3x3xf32>) -> tensor<20x197x395x32xf32> {
    %0 = ttir.empty() : tensor<20x197x395x32xf32>
    // CHECK: = "ttnn.conv_transpose2d"
    // CHECK-SAME: batch_size = 20 : i32
    // CHECK-SAME: dilation = array<i32: 1, 1>
    // CHECK-SAME: groups = 1 : i32
    // CHECK-SAME: in_channels = 16 : i32
    // CHECK-SAME: input_height = 50 : i32
    // CHECK-SAME: input_width = 100 : i32
    // CHECK-SAME: kernel_size = array<i32: 3, 3>
    // CHECK-SAME: out_channels = 32 : i32
    // CHECK-SAME: output_padding = array<i32: 0, 0>
    // CHECK-SAME: padding = array<i32: 1, 2>
    // CHECK-SAME: stride = array<i32: 4, 4>
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %0) <{dilation = array<i32: 1, 1>, groups = 1 : i32, output_padding = array<i32: 0, 0>, padding = array<i32: 1, 2, 1, 2>, stride = array<i32: 4, 4>}> : (tensor<20x50x100x16xf32>, tensor<16x32x3x3xf32>, tensor<20x197x395x32xf32>) -> tensor<20x197x395x32xf32>
    return %1 : tensor<20x197x395x32xf32>
  }
}
