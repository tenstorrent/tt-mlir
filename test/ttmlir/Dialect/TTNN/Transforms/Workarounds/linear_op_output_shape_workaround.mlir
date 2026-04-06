// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test: When the fused kernel path applies (1D or effectively-1D bias, padded
// bias H == TILE_HEIGHT) and the LinearOp output is the broadcasted shape
// (differs from matmul shape), the workaround adjusts the output to matmul
// shape and inserts a reshape.

module {
  // Matmul shape: [256, 512], bias: [1, 1, 512] (effectively 1D).
  // Broadcasted output: [1, 256, 512] != matmul shape [256, 512].
  // Pattern should fire: LinearOp -> [256, 512], then reshape -> [1, 256, 512].
  func.func @linear_fused_kernel_output_shape_adjust(%arg0: tensor<256x1024xbf16>, %arg1: tensor<1024x512xbf16>, %bias: tensor<1x1x512xbf16>) -> tensor<1x256x512xbf16> {
    // CHECK-LABEL: func.func @linear_fused_kernel_output_shape_adjust
    // CHECK: "ttnn.linear"
    // CHECK-SAME: -> tensor<256x512xbf16
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: -> tensor<1x256x512xbf16
    %result = "ttnn.linear"(%arg0, %arg1, %bias) <{transpose_a = false, transpose_b = false}> : (tensor<256x1024xbf16>, tensor<1024x512xbf16>, tensor<1x1x512xbf16>) -> tensor<1x256x512xbf16>
    return %result : tensor<1x256x512xbf16>
  }

  // Matmul shape: [256, 512], bias: [512] (1D).
  // Broadcasted output: [256, 512] == matmul shape.
  // Pattern should NOT fire (shapes already match).
  func.func @linear_fused_kernel_no_change_needed(%arg0: tensor<256x1024xbf16>, %arg1: tensor<1024x512xbf16>, %bias: tensor<512xbf16>) -> tensor<256x512xbf16> {
    // CHECK-LABEL: func.func @linear_fused_kernel_no_change_needed
    // CHECK: "ttnn.linear"
    // CHECK-SAME: -> tensor<256x512xbf16
    // CHECK-NOT: "ttnn.reshape"
    %result = "ttnn.linear"(%arg0, %arg1, %bias) <{transpose_a = false, transpose_b = false}> : (tensor<256x1024xbf16>, tensor<1024x512xbf16>, tensor<512xbf16>) -> tensor<256x512xbf16>
    return %result : tensor<256x512xbf16>
  }

  // Matmul shape: [1, 1, 68, 51200], bias: [1, 68, 51200].
  // Padded bias H = 68 -> 96 != TILE_HEIGHT (32). Composite path.
  // Pattern should NOT fire (not fused kernel path).
  func.func @linear_composite_path_no_change(%arg0: tensor<1x1x68x2048xbf16>, %arg1: tensor<2048x51200xbf16>, %bias: tensor<1x68x51200xbf16>) -> tensor<1x1x68x51200xbf16> {
    // CHECK-LABEL: func.func @linear_composite_path_no_change
    // CHECK-NOT: "ttnn.reshape"
    %result = "ttnn.linear"(%arg0, %arg1, %bias) <{transpose_a = false, transpose_b = false}> : (tensor<1x1x68x2048xbf16>, tensor<2048x51200xbf16>, tensor<1x68x51200xbf16>) -> tensor<1x1x68x51200xbf16>
    return %result : tensor<1x1x68x51200xbf16>
  }

  // Verify that ttcore.weight_dtype discardable attribute is preserved on the
  // new LinearOp after output shape adjustment.
  func.func @linear_output_shape_preserves_weight_dtype(%arg0: tensor<256x1024xbf16>, %arg1: tensor<1024x512xbf16>, %bias: tensor<1x1x512xbf16>) -> tensor<1x256x512xbf16> {
    // CHECK-LABEL: func.func @linear_output_shape_preserves_weight_dtype
    // CHECK: "ttnn.linear"
    // CHECK-SAME: ttcore.weight_dtype = "bfp_bf4"
    // CHECK-SAME: -> tensor<256x512xbf16
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: -> tensor<1x256x512xbf16
    %result = "ttnn.linear"(%arg0, %arg1, %bias) <{transpose_a = false, transpose_b = false}> {ttcore.weight_dtype = "bfp_bf4"} : (tensor<256x1024xbf16>, tensor<1024x512xbf16>, tensor<1x1x512xbf16>) -> tensor<1x256x512xbf16>
    return %result : tensor<1x256x512xbf16>
  }

  // No bias. Pattern should NOT fire.
  func.func @linear_no_bias_no_change(%arg0: tensor<256x1024xbf16>, %arg1: tensor<1024x512xbf16>) -> tensor<256x512xbf16> {
    // CHECK-LABEL: func.func @linear_no_bias_no_change
    // CHECK: "ttnn.linear"
    // CHECK-NOT: "ttnn.reshape"
    %result = "ttnn.linear"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<256x1024xbf16>, tensor<1024x512xbf16>) -> tensor<256x512xbf16>
    return %result : tensor<256x512xbf16>
  }
}
