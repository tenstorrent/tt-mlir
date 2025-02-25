// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s \
// RUN:     --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s

module @max_pool2d attributes {} {
  // Kernel size = 3; Stride = 3; Padding = 1
  func.func @test_maxpool2d_kernel_3x3_stride_3x3_padding_1(%arg0: tensor<1x128x32x32xbf16>) -> tensor<1x128x11x11xbf16> {
    // CHECK-LABEL: func.func @test_maxpool2d_kernel_3x3_stride_3x3_padding_1(
    %cst = stablehlo.constant dense<0xFF80> : tensor<bf16>
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttnn.permute"(%arg0)
    // CHECK-SAME: permutation = array<i64: 0, 2, 3, 1>
    // CHECK-SAME: tensor<1x128x32x32xbf16
    // CHECK-SAME: -> tensor<1x32x32x128xbf16
    // CHECK: "ttnn.reshape"(%[[PERMUTE]])
    // CHECK-SAME: shape = [1 : i32, 1 : i32, 1024 : i32, 128 : i32]
    // CHECK-SAME: tensor<1x32x32x128xbf16
    // CHECK-SAME: -> tensor<1x1x1024x128xbf16
    // CHECK: "ttnn.max_pool2d"
    // CHECK-SAME: batch_size = 1 : si32,
    // CHECK-SAME: ceil_mode = false,
    // CHECK-SAME: channels = 128 : si32,
    // CHECK-SAME: dilation_height = 1 : si32, dilation_width = 1 : si32,
    // CHECK-SAME: input_height = 32 : si32, input_width = 32 : si32,
    // CHECK-SAME: kernel_height = 3 : si32, kernel_width = 3 : si32,
    // CHECK-SAME: padding_height = 1 : si32, padding_width = 1 : si32,
    // CHECK-SAME: stride_height = 3 : si32, stride_width = 3 : si32}
    // CHECK-SAME: tensor<1x1x1024x128xbf16
    // CHECK-SAME: tensor<1x1x121x128xbf16
    // CHECK-SAME: -> tensor<1x1x121x128xbf16
    %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 3, 3>}> ({
    ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
      %1 = stablehlo.maximum %arg1, %arg2 : tensor<bf16>
      stablehlo.return %1 : tensor<bf16>
    }) : (tensor<1x128x32x32xbf16>, tensor<bf16>) -> tensor<1x128x11x11xbf16>
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 11 : i32, 11 : i32, 128 : i32]
    // CHECK-SAME: tensor<1x1x121x128xbf16
    // CHECK-SAME: -> tensor<1x11x11x128xbf16
    // CHECK: %[[RET:[0-9]+]] = "ttnn.permute"(%[[RESHAPE]])
    // CHECK-SAME: permutation = array<i64: 0, 3, 1, 2>
    // CHECK-SAME: tensor<1x11x11x128xbf16
    // CHECK-SAME: -> tensor<1x128x11x11xbf16
    // CHECK: return %[[RET]] : tensor<1x128x11x11xbf16
    return %0 : tensor<1x128x11x11xbf16>
  }

  // Kernel size = 1; Stride = 1; Padding = 0
  func.func @test_maxpool2d_kernel_1x1_stride_1x1_padding_0(%arg0: tensor<1x192x28x28xbf16>) -> tensor<1x192x28x28xbf16> {
    // CHECK-LABEL: func.func @test_maxpool2d_kernel_1x1_stride_1x1_padding_0(
    %cst = stablehlo.constant dense<0xFF80> : tensor<bf16>
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttnn.permute"(%arg0)
    // CHECK-SAME: permutation = array<i64: 0, 2, 3, 1>
    // CHECK-SAME: tensor<1x192x28x28xbf16
    // CHECK-SAME: -> tensor<1x28x28x192xbf16
    // CHECK: "ttnn.reshape"(%[[PERMUTE]])
    // CHECK-SAME: shape = [1 : i32, 1 : i32, 784 : i32, 192 : i32]
    // CHECK-SAME: tensor<1x28x28x192xbf16
    // CHECK-SAME: -> tensor<1x1x784x192xbf16
    // CHECK: "ttnn.max_pool2d"
    // CHECK-SAME: batch_size = 1 : si32,
    // CHECK-SAME: ceil_mode = false,
    // CHECK-SAME: channels = 192 : si32,
    // CHECK-SAME: dilation_height = 1 : si32, dilation_width = 1 : si32,
    // CHECK-SAME: input_height = 28 : si32, input_width = 28 : si32,
    // CHECK-SAME: kernel_height = 1 : si32, kernel_width = 1 : si32,
    // CHECK-SAME: padding_height = 0 : si32, padding_width = 0 : si32,
    // CHECK-SAME: stride_height = 1 : si32, stride_width = 1 : si32}
    // CHECK-SAME: tensor<1x1x784x192xbf16
    // CHECK-SAME: tensor<1x1x784x192xbf16
    // CHECK-SAME: -> tensor<1x1x784x192xbf16
    %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1>, window_strides = array<i64: 1, 1, 1, 1>}> ({
    ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
      %1 = stablehlo.maximum %arg1, %arg2 : tensor<bf16>
      stablehlo.return %1 : tensor<bf16>
    }) : (tensor<1x192x28x28xbf16>, tensor<bf16>) -> tensor<1x192x28x28xbf16>
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 28 : i32, 28 : i32, 192 : i32]
    // CHECK-SAME: tensor<1x1x784x192xbf16
    // CHECK-SAME: -> tensor<1x28x28x192xbf16
    // CHECK: %[[RET:[0-9]+]] = "ttnn.permute"(%[[RESHAPE]])
    // CHECK-SAME: permutation = array<i64: 0, 3, 1, 2>
    // CHECK-SAME: tensor<1x28x28x192xbf16
    // CHECK-SAME: -> tensor<1x192x28x28xbf16
    // CHECK: return %[[RET]] : tensor<1x192x28x28xbf16
    return %0 : tensor<1x192x28x28xbf16>
  }

  // Kernel size = (1, 2) ; Stride = (3, 1); Padding = 0
  func.func @test_maxpool2d_kernel_1x2_stride_3x1_padding_0(%arg0: tensor<1x192x28x28xbf16>) -> tensor<1x192x10x27xbf16> {
    // CHECK-LABEL: func.func @test_maxpool2d_kernel_1x2_stride_3x1_padding_0(
    %cst = stablehlo.constant dense<0xFF80> : tensor<bf16>
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttnn.permute"(%arg0)
    // CHECK-SAME: permutation = array<i64: 0, 2, 3, 1>
    // CHECK-SAME: tensor<1x192x28x28xbf16
    // CHECK-SAME: -> tensor<1x28x28x192xbf16
    // CHECK: "ttnn.reshape"(%[[PERMUTE]])
    // CHECK-SAME: shape = [1 : i32, 1 : i32, 784 : i32, 192 : i32]
    // CHECK-SAME: tensor<1x28x28x192xbf16
    // CHECK-SAME: -> tensor<1x1x784x192xbf16
    // CHECK: "ttnn.max_pool2d"
    // CHECK-SAME: batch_size = 1 : si32,
    // CHECK-SAME: ceil_mode = false,
    // CHECK-SAME: channels = 192 : si32,
    // CHECK-SAME: dilation_height = 1 : si32, dilation_width = 1 : si32,
    // CHECK-SAME: input_height = 28 : si32, input_width = 28 : si32,
    // CHECK-SAME: kernel_height = 1 : si32, kernel_width = 2 : si32,
    // CHECK-SAME: padding_height = 0 : si32, padding_width = 0 : si32,
    // CHECK-SAME: stride_height = 3 : si32, stride_width = 1 : si32}
    // CHECK-SAME: tensor<1x1x784x192xbf16
    // CHECK-SAME: tensor<1x1x270x192xbf16
    // CHECK-SAME: -> tensor<1x1x270x192xbf16
    %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 2>, window_strides = array<i64: 1, 1, 3, 1>}> ({
    ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
      %1 = stablehlo.maximum %arg1, %arg2 : tensor<bf16>
      stablehlo.return %1 : tensor<bf16>
    }) : (tensor<1x192x28x28xbf16>, tensor<bf16>) -> tensor<1x192x10x27xbf16>
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 10 : i32, 27 : i32, 192 : i32]
    // CHECK-SAME: tensor<1x1x270x192xbf16
    // CHECK-SAME: -> tensor<1x10x27x192xbf16
    // CHECK: %[[RET:[0-9]+]] = "ttnn.permute"(%[[RESHAPE]])
    // CHECK-SAME: permutation = array<i64: 0, 3, 1, 2>
    // CHECK-SAME: tensor<1x10x27x192xbf16
    // CHECK-SAME: -> tensor<1x192x10x27xbf16
    // CHECK: return %[[RET]] : tensor<1x192x10x27xbf16
    return %0 : tensor<1x192x10x27xbf16>
  }
}
