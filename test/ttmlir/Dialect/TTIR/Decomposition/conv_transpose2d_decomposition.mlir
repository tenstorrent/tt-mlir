// RUN: ttmlir-opt --ttir-to-ttir-decomposition --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

module @test_conv_transpose {
  func.func @test_conv_transpose2d(%arg0: tensor<1x256x32x32xf32>, %arg1: tensor<256x128x2x2xf32>, %arg2: tensor<128xf32>) -> tensor<1x128x64x64xf32> {
    // CHECK-LABEL: @test_conv_transpose2d
    // CHECK-NOT: ttir.reverse
    %0 = ttir.empty() : tensor<2x2x128x256xf32>
    %1 = "ttir.permute"(%arg1, %0) <{permutation = array<i64: 2, 3, 1, 0>}> : (tensor<256x128x2x2xf32>, tensor<2x2x128x256xf32>) -> tensor<2x2x128x256xf32>
    %2 = ttir.empty() : tensor<2x2x128x256xf32>
    %3 = "ttir.reverse"(%1, %2) <{dimensions = array<i64: 0, 1>}> : (tensor<2x2x128x256xf32>, tensor<2x2x128x256xf32>) -> tensor<2x2x128x256xf32>
    %4 = ttir.empty() : tensor<1x128x64x64xf32>
    // CHECK: %[[ARG0:[0-9]+]] = "ttir.permute"(%arg0, %{{[0-9]+}})
    // CHECK-SAME: <{permutation = array<i64: 0, 2, 3, 1>}>
    // CHECK-SAME: (tensor<1x256x32x32xf32>, tensor<1x32x32x256xf32>)
    // CHECK-SAME: -> tensor<1x32x32x256xf32>
    // CHECK: %[[CONV_T:[0-9]+]] = "ttir.conv_transpose2d"(%[[ARG0]], %{{[0-9]+}}, %{{[0-9]+}})
    // CHECK-SAME: dilation = array<i32: 1, 1>,
    // CHECK-SAME: groups = 1 : i32,
    // CHECK-SAME: output_padding = array<i32: 0, 0>,
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0>,
    // CHECK-SAME: stride = array<i32: 2, 2>
    // CHECK-SAME: (tensor<1x32x32x256xf32>, tensor<256x128x2x2xf32>, tensor<1x64x64x128xf32>)
    // CHECK-SAME: -> tensor<1x64x64x128xf32>
    %5 = "ttir.convolution"(%arg0, %3, %4) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 2, kernel_input_feature = 3, kernel_spatial_dimensions = 0x1, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 2, 2>, is_transposed = true, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x256x32x32xf32>, tensor<2x2x128x256xf32>, tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    // CHECK: %{{[0-9]+}} = "ttir.permute"(%[[CONV_T]], %{{[0-9]+}})
    // CHECK-SAME: permutation = array<i64: 0, 3, 1, 2>
    // CHECK-SAME: (tensor<1x64x64x128xf32>, tensor<1x128x64x64xf32>)
    // CHECK-SAME: -> tensor<1x128x64x64xf32>
    %6 = ttir.empty() : tensor<128x1x1xf32>
    %7 = "ttir.reshape"(%arg2, %6) <{shape = [128 : i32, 1 : i32, 1 : i32]}> : (tensor<128xf32>, tensor<128x1x1xf32>) -> tensor<128x1x1xf32>
    %8 = ttir.empty() : tensor<1x128x64x64xf32>
    %9 = "ttir.broadcast"(%5, %8) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x128x64x64xf32>, tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %10 = ttir.empty() : tensor<1x128x1x1xf32>
    %11 = "ttir.reshape"(%7, %10) <{shape = [1 : i32, 128 : i32, 1 : i32, 1 : i32]}> : (tensor<128x1x1xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x1x1xf32>
    %12 = ttir.empty() : tensor<1x128x64x64xf32>
    %13 = "ttir.broadcast"(%11, %12) <{broadcast_dimensions = array<i64: 1, 1, 64, 64>}> : (tensor<1x128x1x1xf32>, tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    %14 = ttir.empty() : tensor<1x128x64x64xf32>
    %15 = "ttir.add"(%9, %13, %14) : (tensor<1x128x64x64xf32>, tensor<1x128x64x64xf32>, tensor<1x128x64x64xf32>) -> tensor<1x128x64x64xf32>
    return %15 : tensor<1x128x64x64xf32>
  }
}
