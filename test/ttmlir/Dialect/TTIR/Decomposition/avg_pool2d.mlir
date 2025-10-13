// RUN: ttmlir-opt --ttir-to-ttir-decomposition -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  // Kernel size = 1; stride = 1 with padding (to prevent folding)
  func.func @test_avgpool2d_kernel_1x1_stride_1x1(%arg0: tensor<1x192x28x28xbf16>) -> tensor<1x192x30x30xbf16> {
    // CHECK-LABEL: func.func @test_avgpool2d_kernel_1x1_stride_1x1(
    %0 = ttir.empty() : tensor<1x192x30x30xbf16>
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%arg0
    // CHECK-SAME: permutation = array<i64: 0, 2, 3, 1>
    // CHECK-SAME: (tensor<1x192x28x28xbf16>, tensor<1x28x28x192xbf16>)
    // CHECK-SAME: -> tensor<1x28x28x192xbf16>
    // CHECK: %[[AVGPOOL:[0-9]+]] = "ttir.avg_pool2d"(%[[PERMUTE]],
    // CHECK-SAME: ceil_mode = false
    // CHECK-SAME: dilation = array<i32: 1, 1>
    // CHECK-SAME: kernel = array<i32: 1, 1>
    // CHECK-SAME: padding = array<i32: 1, 1, 1, 1>
    // CHECK-SAME: stride = array<i32: 1, 1>
    // CHECK-SAME: (tensor<1x28x28x192xbf16>, tensor<1x30x30x192xbf16>)
    // CHECK-SAME: -> tensor<1x30x30x192xbf16>
    %1 = "ttir.pooling"(%arg0, %0) <{base_dilations = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1>, padding = array<i64: 0, 0, 0, 0, 1, 1, 1, 1>, pooling_method = #ttir<pooling_method Average>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1>, window_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x192x28x28xbf16>, tensor<1x192x30x30xbf16>) -> tensor<1x192x30x30xbf16>
    // CHECK: %[[RET:[0-9]+]] = "ttir.permute"(%[[AVGPOOL]],
    // CHECK-SAME: permutation = array<i64: 0, 3, 1, 2>
    // CHECK-SAME: (tensor<1x30x30x192xbf16>, tensor<1x192x30x30xbf16>)
    // CHECK-SAME: -> tensor<1x192x30x30xbf16>
    // CHECK: return %[[RET]] : tensor<1x192x30x30xbf16>
    return %1 : tensor<1x192x30x30xbf16>
  }

  // Kernel size = 3; stride = 1
  func.func @test_avgpool2d_kernel_3x3_stride_1x1(%arg0: tensor<1x256x28x28xbf16>) -> tensor<1x256x28x28xbf16> {
    // CHECK-LABEL: func.func @test_avgpool2d_kernel_3x3_stride_1x1(
    %0 = ttir.empty() : tensor<1x256x28x28xbf16>
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%arg0
    // CHECK-SAME: permutation = array<i64: 0, 2, 3, 1>
    // CHECK-SAME: (tensor<1x256x28x28xbf16>, tensor<1x28x28x256xbf16>)
    // CHECK-SAME: -> tensor<1x28x28x256xbf16>
    // CHECK: %[[AVGPOOL:[0-9]+]] = "ttir.avg_pool2d"(%[[PERMUTE]],
    // CHECK-SAME: ceil_mode = false
    // CHECK-SAME: dilation = array<i32: 1, 1>
    // CHECK-SAME: kernel = array<i32: 3, 3>
    // CHECK-SAME: padding = array<i32: 1, 1, 1, 1>
    // CHECK-SAME: stride = array<i32: 1, 1>
    // CHECK-SAME: (tensor<1x28x28x256xbf16>, tensor<1x28x28x256xbf16>)
    // CHECK-SAME: -> tensor<1x28x28x256xbf16>
    %1 = "ttir.pooling"(%arg0, %0) <{base_dilations = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1>, padding = array<i64: 0, 0, 0, 0, 1, 1, 1, 1>, pooling_method = #ttir<pooling_method Average>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x256x28x28xbf16>, tensor<1x256x28x28xbf16>) -> tensor<1x256x28x28xbf16>
    // CHECK: %[[RET:[0-9]+]] = "ttir.permute"(%[[AVGPOOL]],
    // CHECK-SAME: permutation = array<i64: 0, 3, 1, 2>
    // CHECK-SAME: (tensor<1x28x28x256xbf16>, tensor<1x256x28x28xbf16>)
    // CHECK-SAME: -> tensor<1x256x28x28xbf16>
    // CHECK: return %[[RET]] : tensor<1x256x28x28xbf16>
    return %1 : tensor<1x256x28x28xbf16>
  }

  // Kernel size = (2, 1); stride = 1
  func.func @test_avgpool2d_kernel_2x1_stride_1x1(%arg0: tensor<1x192x28x28xbf16>) -> tensor<1x192x27x28xbf16> {
    // CHECK-LABEL: func.func @test_avgpool2d_kernel_2x1_stride_1x1(
    %0 = ttir.empty() : tensor<1x192x27x28xbf16>
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%arg0
    // CHECK-SAME: permutation = array<i64: 0, 2, 3, 1>
    // CHECK-SAME: (tensor<1x192x28x28xbf16>, tensor<1x28x28x192xbf16>)
    // CHECK-SAME: -> tensor<1x28x28x192xbf16>
    // CHECK: %[[AVGPOOL:[0-9]+]] = "ttir.avg_pool2d"(%[[PERMUTE]],
    // CHECK-SAME: ceil_mode = false,
    // CHECK-SAME: dilation = array<i32: 1, 1>
    // CHECK-SAME: kernel = array<i32: 2, 1>
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0>
    // CHECK-SAME: stride = array<i32: 1, 1>
    // CHECK-SAME: (tensor<1x28x28x192xbf16>, tensor<1x27x28x192xbf16>)
    // CHECK-SAME: -> tensor<1x27x28x192xbf16>
    %1 = "ttir.pooling"(%arg0, %0) <{base_dilations = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1>, padding = array<i64: 0, 0, 0, 0, 0, 0, 0, 0>, pooling_method = #ttir<pooling_method Average>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 2, 1>, window_strides = array<i64: 1, 1, 1, 1>}> : (tensor<1x192x28x28xbf16>, tensor<1x192x27x28xbf16>) -> tensor<1x192x27x28xbf16>
    // CHECK: %[[RET:[0-9]+]] = "ttir.permute"(%[[AVGPOOL]],
    // CHECK-SAME: permutation = array<i64: 0, 3, 1, 2>
    // CHECK-SAME: (tensor<1x27x28x192xbf16>, tensor<1x192x27x28xbf16>)
    // CHECK-SAME: -> tensor<1x192x27x28xbf16>
    // CHECK: return %[[RET]] : tensor<1x192x27x28xbf16>
    return %1 : tensor<1x192x27x28xbf16>
  }

  // Kernel size = (1, 2); stride = (3, 1)
  func.func @test_avgpool2d_kernel_1x2_stride_3x1(%arg0: tensor<1x192x28x28xbf16>) -> tensor<1x192x10x27xbf16> {
    // CHECK-LABEL: func.func @test_avgpool2d_kernel_1x2_stride_3x1(
    %0 = ttir.empty() : tensor<1x192x10x27xbf16>
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%arg0
    // CHECK-SAME: permutation = array<i64: 0, 2, 3, 1>
    // CHECK-SAME: (tensor<1x192x28x28xbf16>, tensor<1x28x28x192xbf16>)
    // CHECK-SAME: -> tensor<1x28x28x192xbf16>
    // CHECK: %[[AVGPOOL:[0-9]+]] = "ttir.avg_pool2d"(%[[PERMUTE]],
    // CHECK-SAME: ceil_mode = false,
    // CHECK-SAME: dilation = array<i32: 1, 1>
    // CHECK-SAME: kernel = array<i32: 1, 2>
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0>
    // CHECK-SAME: stride = array<i32: 3, 1>
    // CHECK-SAME: (tensor<1x28x28x192xbf16>, tensor<1x10x27x192xbf16>)
    // CHECK-SAME: -> tensor<1x10x27x192xbf16>
    %1 = "ttir.pooling"(%arg0, %0) <{base_dilations = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1>, padding = array<i64: 0, 0, 0, 0, 0, 0, 0, 0>, pooling_method = #ttir<pooling_method Average>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 2>, window_strides = array<i64: 1, 1, 3, 1>}> : (tensor<1x192x28x28xbf16>, tensor<1x192x10x27xbf16>) -> tensor<1x192x10x27xbf16>
    // CHECK: %[[RET:[0-9]+]] = "ttir.permute"(%[[AVGPOOL]],
    // CHECK-SAME: permutation = array<i64: 0, 3, 1, 2>
    // CHECK-SAME: (tensor<1x10x27x192xbf16>, tensor<1x192x10x27xbf16>)
    // CHECK-SAME: -> tensor<1x192x10x27xbf16>
    // CHECK: return %[[RET]] : tensor<1x192x10x27xbf16>
    return %1 : tensor<1x192x10x27xbf16>
  }

  // Test 2D to 4D reshaping - Kernel size = 2x2; stride = 2x2
  func.func @test_avgpool2d_2d_input_kernel_2x2_stride_2x2(%arg0: tensor<28x28xbf16>) -> tensor<14x14xbf16> {
    // CHECK-LABEL: func.func @test_avgpool2d_2d_input_kernel_2x2_stride_2x2(
    %0 = ttir.empty() : tensor<14x14xbf16>
    // CHECK: %[[RESHAPE_INPUT:[0-9]+]] = "ttir.reshape"(%arg0
    // CHECK-SAME: shape = array<i32: 1, 1, 28, 28>
    // CHECK-SAME: (tensor<28x28xbf16>, tensor<1x1x28x28xbf16>)
    // CHECK-SAME: -> tensor<1x1x28x28xbf16>
    // CHECK: %[[RESHAPE_OUTPUT:[0-9]+]] = "ttir.reshape"(
    // CHECK-SAME: shape = array<i32: 1, 1, 14, 14>
    // CHECK-SAME: (tensor<14x14xbf16>, tensor<1x1x14x14xbf16>)
    // CHECK-SAME: -> tensor<1x1x14x14xbf16>
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%[[RESHAPE_INPUT]]
    // CHECK-SAME: permutation = array<i64: 0, 2, 3, 1>
    // CHECK-SAME: (tensor<1x1x28x28xbf16>, tensor<1x28x28x1xbf16>)
    // CHECK-SAME: -> tensor<1x28x28x1xbf16>
    // CHECK: %[[AVGPOOL:[0-9]+]] = "ttir.avg_pool2d"(%[[PERMUTE]],
    // CHECK-SAME: ceil_mode = false
    // CHECK-SAME: dilation = array<i32: 1, 1>
    // CHECK-SAME: kernel = array<i32: 2, 2>
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0>
    // CHECK-SAME: stride = array<i32: 2, 2>
    // CHECK-SAME: (tensor<1x28x28x1xbf16>, tensor<1x14x14x1xbf16>)
    // CHECK-SAME: -> tensor<1x14x14x1xbf16>
    %1 = "ttir.pooling"(%arg0, %0) <{base_dilations = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 1>, padding = array<i64: 0, 0, 0, 0>, pooling_method = #ttir<pooling_method Average>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 2, 2>, window_strides = array<i64: 2, 2>}> : (tensor<28x28xbf16>, tensor<14x14xbf16>) -> tensor<14x14xbf16>
    // CHECK: %[[PERMUTE_BACK:[0-9]+]] = "ttir.permute"(%[[AVGPOOL]],
    // CHECK-SAME: permutation = array<i64: 0, 3, 1, 2>
    // CHECK-SAME: (tensor<1x14x14x1xbf16>, tensor<1x1x14x14xbf16>)
    // CHECK-SAME: -> tensor<1x1x14x14xbf16>
    // CHECK: %[[RESHAPE_BACK:[0-9]+]] = "ttir.reshape"(%[[PERMUTE_BACK]]
    // CHECK-SAME: shape = array<i32: 14, 14>
    // CHECK-SAME: (tensor<1x1x14x14xbf16>, tensor<14x14xbf16>)
    // CHECK-SAME: -> tensor<14x14xbf16>
    // CHECK: return %[[RESHAPE_BACK]] : tensor<14x14xbf16>
    return %1 : tensor<14x14xbf16>
  }
}
