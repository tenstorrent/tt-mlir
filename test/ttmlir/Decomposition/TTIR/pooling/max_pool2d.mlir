// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s | FileCheck %s

module attributes {} {
  // Kernel size = 1; stride = 1
  func.func @test_maxpool2d_kernel_1x1_stride_1x1(%arg0: tensor<1x192x28x28xbf16>) -> tensor<1x192x28x28xbf16> {
    // CHECK-LABEL: func.func @test_maxpool2d_kernel_1x1_stride_1x1(
    %0 = tensor.empty() : tensor<1x192x28x28xbf16>
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%arg0
    // CHECK-SAME: permutation = array<i64: 0, 2, 3, 1>
    // CHECK-SAME: (tensor<1x192x28x28xbf16>, tensor<1x28x28x192xbf16>)
    // CHECK-SAME: -> tensor<1x28x28x192xbf16>
    // CHECK: %[[MAXPOOL:[0-9]+]] = "ttir.max_pool2d"(%[[PERMUTE]],
    // CHECK-SAME: ceil_mode = false,
    // CHECK-SAME: dilation_height = 1 : si32, dilation_width = 1 : si32,
    // CHECK-SAME: kernel_height = 1 : si32, kernel_width = 1 : si32,
    // CHECK-SAME: padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32,
    // CHECK-SAME: stride_height = 1 : si32, stride_width = 1 : si32
    // CHECK-SAME: (tensor<1x28x28x192xbf16>, tensor<1x28x28x192xbf16>)
    // CHECK-SAME: -> tensor<1x28x28x192xbf16>
    %1 = "ttir.pooling"(%arg0, %0) <{base_dilations = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1>, padding = array<i64: 0, 0, 0, 0, 0, 0, 0, 0>, pooling_method = #ttir<pooling_method Max>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1>, window_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x192x28x28xbf16>, tensor<1x192x28x28xbf16>) -> tensor<1x192x28x28xbf16>
    // CHECK: %[[RET:[0-9]+]] = "ttir.permute"(%[[MAXPOOL]],
    // CHECK-SAME: permutation = array<i64: 0, 3, 1, 2>
    // CHECK-SAME: (tensor<1x28x28x192xbf16>, tensor<1x192x28x28xbf16>)
    // CHECK-SAME: -> tensor<1x192x28x28xbf16>
    // CHECK: return %[[RET]] : tensor<1x192x28x28xbf16>
    return %1 : tensor<1x192x28x28xbf16>
  }

  // Kernel size = 3; stride = 1
  func.func @test_maxpool2d_kernel_3x3_stride_1x1(%arg0: tensor<1x256x28x28xbf16>) -> tensor<1x256x28x28xbf16> {
    // CHECK-LABEL: func.func @test_maxpool2d_kernel_3x3_stride_1x1(
    %0 = tensor.empty() : tensor<1x256x28x28xbf16>
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%arg0
    // CHECK-SAME: permutation = array<i64: 0, 2, 3, 1>
    // CHECK-SAME: (tensor<1x256x28x28xbf16>, tensor<1x28x28x256xbf16>)
    // CHECK-SAME: -> tensor<1x28x28x256xbf16>
    // CHECK: %[[MAXPOOL:[0-9]+]] = "ttir.max_pool2d"(%[[PERMUTE]],
    // CHECK-SAME: ceil_mode = false,
    // CHECK-SAME: dilation_height = 1 : si32, dilation_width = 1 : si32,
    // CHECK-SAME: kernel_height = 3 : si32, kernel_width = 3 : si32,
    // CHECK-SAME: padding_bottom = 1 : si32, padding_left = 1 : si32, padding_right = 1 : si32, padding_top = 1 : si32,
    // CHECK-SAME: stride_height = 1 : si32, stride_width = 1 : si32
    // CHECK-SAME: (tensor<1x28x28x256xbf16>, tensor<1x28x28x256xbf16>)
    // CHECK-SAME: -> tensor<1x28x28x256xbf16>
    %1 = "ttir.pooling"(%arg0, %0) <{base_dilations = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1>, padding = array<i64: 0, 0, 0, 0, 1, 1, 1, 1>, pooling_method = #ttir<pooling_method Max>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x256x28x28xbf16>, tensor<1x256x28x28xbf16>) -> tensor<1x256x28x28xbf16>
    // CHECK: %[[RET:[0-9]+]] = "ttir.permute"(%[[MAXPOOL]],
    // CHECK-SAME: permutation = array<i64: 0, 3, 1, 2>
    // CHECK-SAME: (tensor<1x28x28x256xbf16>, tensor<1x256x28x28xbf16>)
    // CHECK-SAME: -> tensor<1x256x28x28xbf16>
    // CHECK: return %[[RET]] : tensor<1x256x28x28xbf16>
    return %1 : tensor<1x256x28x28xbf16>
  }

  // Kernel size = (2, 1); stride = 1
  func.func @test_maxpool2d_kernel_2x1_stride_1x1(%arg0: tensor<1x192x28x28xbf16>) -> tensor<1x192x27x28xbf16> {
    // CHECK-LABEL: func.func @test_maxpool2d_kernel_2x1_stride_1x1(
    %0 = tensor.empty() : tensor<1x192x27x28xbf16>
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%arg0
    // CHECK-SAME: permutation = array<i64: 0, 2, 3, 1>
    // CHECK-SAME: (tensor<1x192x28x28xbf16>, tensor<1x28x28x192xbf16>)
    // CHECK-SAME: -> tensor<1x28x28x192xbf16>
    // CHECK: %[[MAXPOOL:[0-9]+]] = "ttir.max_pool2d"(%[[PERMUTE]],
    // CHECK-SAME: ceil_mode = false,
    // CHECK-SAME: dilation_height = 1 : si32, dilation_width = 1 : si32,
    // CHECK-SAME: kernel_height = 2 : si32, kernel_width = 1 : si32,
    // CHECK-SAME: padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32,
    // CHECK-SAME: stride_height = 1 : si32, stride_width = 1 : si32
    // CHECK-SAME: (tensor<1x28x28x192xbf16>, tensor<1x27x28x192xbf16>)
    // CHECK-SAME: -> tensor<1x27x28x192xbf16>
    %1 = "ttir.pooling"(%arg0, %0) <{base_dilations = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1>, padding = array<i64: 0, 0, 0, 0, 0, 0, 0, 0>, pooling_method = #ttir<pooling_method Max>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 2, 1>, window_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x192x28x28xbf16>, tensor<1x192x27x28xbf16>) -> tensor<1x192x27x28xbf16>
    // CHECK: %[[RET:[0-9]+]] = "ttir.permute"(%[[MAXPOOL]],
    // CHECK-SAME: permutation = array<i64: 0, 3, 1, 2>
    // CHECK-SAME: (tensor<1x27x28x192xbf16>, tensor<1x192x27x28xbf16>)
    // CHECK-SAME: -> tensor<1x192x27x28xbf16>
    // CHECK: return %[[RET]] : tensor<1x192x27x28xbf16>
    return %1 : tensor<1x192x27x28xbf16>
  }

  // Kernel size = (1, 2); stride = (3, 1)
  func.func @test_maxpool2d_kernel_1x2_stride_3x1(%arg0: tensor<1x192x28x28xbf16>) -> tensor<1x192x10x27xbf16> {
    // CHECK-LABEL: func.func @test_maxpool2d_kernel_1x2_stride_3x1(
    %0 = tensor.empty() : tensor<1x192x10x27xbf16>
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%arg0
    // CHECK-SAME: permutation = array<i64: 0, 2, 3, 1>
    // CHECK-SAME: (tensor<1x192x28x28xbf16>, tensor<1x28x28x192xbf16>)
    // CHECK-SAME: -> tensor<1x28x28x192xbf16>
    // CHECK: %[[MAXPOOL:[0-9]+]] = "ttir.max_pool2d"(%[[PERMUTE]],
    // CHECK-SAME: ceil_mode = false,
    // CHECK-SAME: dilation_height = 1 : si32, dilation_width = 1 : si32,
    // CHECK-SAME: kernel_height = 1 : si32, kernel_width = 2 : si32,
    // CHECK-SAME: padding_bottom = 0 : si32, padding_left = 0 : si32, padding_right = 0 : si32, padding_top = 0 : si32,
    // CHECK-SAME: stride_height = 3 : si32, stride_width = 1 : si32
    // CHECK-SAME: (tensor<1x28x28x192xbf16>, tensor<1x10x27x192xbf16>)
    // CHECK-SAME: -> tensor<1x10x27x192xbf16>
    %1 = "ttir.pooling"(%arg0, %0) <{base_dilations = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1>, padding = array<i64: 0, 0, 0, 0, 0, 0, 0, 0>, pooling_method = #ttir<pooling_method Max>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 2>, window_strides = array<i64: 1, 1, 3, 1>}> : (tensor<1x192x28x28xbf16>, tensor<1x192x10x27xbf16>) -> tensor<1x192x10x27xbf16>
    // CHECK: %[[RET:[0-9]+]] = "ttir.permute"(%[[MAXPOOL]],
    // CHECK-SAME: permutation = array<i64: 0, 3, 1, 2>
    // CHECK-SAME: (tensor<1x10x27x192xbf16>, tensor<1x192x10x27xbf16>)
    // CHECK-SAME: -> tensor<1x192x10x27xbf16>
    // CHECK: return %[[RET]] : tensor<1x192x10x27xbf16>
    return %1 : tensor<1x192x10x27xbf16>
  }
}
