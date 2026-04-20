// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

module  {
  func.func @linear_with_batched_rhs_and_bias(%arg0: tensor<2x33x1024xf32>, %arg1: tensor<2x1024x1024xf32>, %arg2: tensor<2x33x1024xf32>) -> tensor<2x33x1024xf32>{
    // CHECK-LABEL: func.func @linear_with_batched_rhs_and_bias
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: -> tensor<2x33x1024xf32
    // CHECK: "ttnn.add"
    // CHECK-SAME: -> tensor<2x33x1024xf32
    %result = "ttnn.linear"(%arg0, %arg1, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<2x33x1024xf32>, tensor<2x1024x1024xf32>, tensor<2x33x1024xf32>) -> tensor<2x33x1024xf32>
    return %result : tensor<2x33x1024xf32>
  }

  func.func @linear_bias_broadcast(%arg0: tensor<4x3x64x128xbf16>, %arg1: tensor<4x3x128x32xbf16>, %bias: tensor<14x4x3x64x32xbf16>) -> tensor<14x4x3x64x32xbf16> {
    // CHECK-LABEL: func.func @linear_bias_broadcast
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: -> tensor<4x3x64x32xbf16
    // CHECK: "ttnn.add"
    // CHECK-SAME: -> tensor<14x4x3x64x32xbf16
    %result = "ttnn.linear"(%arg0, %arg1, %bias) : (tensor<4x3x64x128xbf16>, tensor<4x3x128x32xbf16>, tensor<14x4x3x64x32xbf16>) -> tensor<14x4x3x64x32xbf16>
    return %result : tensor<14x4x3x64x32xbf16>
  }

  func.func @linear_nd_nd_bias_broadcast_matmul(%arg0: tensor<1x3x64x128xbf16>, %arg1: tensor<1x3x128x32xbf16>, %bias: tensor<14x4x3x64x32xbf16>) -> tensor<14x4x3x64x32xbf16> {
    // CHECK-LABEL: func.func @linear_nd_nd_bias_broadcast_matmul
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: -> tensor<1x3x64x32xbf16
    // CHECK: "ttnn.add"
    // CHECK-SAME: -> tensor<14x4x3x64x32xbf16
    %result = "ttnn.linear"(%arg0, %arg1, %bias) : (tensor<1x3x64x128xbf16>, tensor<1x3x128x32xbf16>, tensor<14x4x3x64x32xbf16>) -> tensor<14x4x3x64x32xbf16>
    return %result : tensor<14x4x3x64x32xbf16>
  }
  func.func @linear_with_sigmoid(%arg0: tensor<100x384xbf16>, %arg1: tensor<4x384xbf16>, %arg2: tensor<1x100x4xbf16>) -> tensor<1x100x4xbf16> {
    // CHECK-LABEL: func.func @linear_with_sigmoid
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: -> tensor<100x4xbf16
    // CHECK: "ttnn.add"
    // CHECK-SAME: -> tensor<1x100x4xbf16
    // CHECK: "ttnn.sigmoid"
    // CHECK-SAME: -> tensor<1x100x4xbf16
    // CHECK-NOT: "ttnn.linear"
    // CHECK: return
    %result = "ttnn.linear"(%arg0, %arg1, %arg2) <{activation = "sigmoid", transpose_a = false, transpose_b = true}> : (tensor<100x384xbf16>, tensor<4x384xbf16>, tensor<1x100x4xbf16>) -> tensor<1x100x4xbf16>
    return %result : tensor<1x100x4xbf16>
  }
  func.func @linear_with_relu(%arg0: tensor<100x384xbf16>, %arg1: tensor<4x384xbf16>, %arg2: tensor<1x100x4xbf16>) -> tensor<1x100x4xbf16> {
    // CHECK-LABEL: func.func @linear_with_relu
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: -> tensor<100x4xbf16
    // CHECK: "ttnn.add"
    // CHECK-SAME: -> tensor<1x100x4xbf16
    // CHECK: "ttnn.relu"
    // CHECK-SAME: -> tensor<1x100x4xbf16
    // CHECK-NOT: "ttnn.linear"
    // CHECK: return
    %result = "ttnn.linear"(%arg0, %arg1, %arg2) <{activation = "relu", transpose_a = false, transpose_b = true}> : (tensor<100x384xbf16>, tensor<4x384xbf16>, tensor<1x100x4xbf16>) -> tensor<1x100x4xbf16>
    return %result : tensor<1x100x4xbf16>
  }
  func.func @linear_with_gelu(%arg0: tensor<100x384xbf16>, %arg1: tensor<4x384xbf16>, %arg2: tensor<1x100x4xbf16>) -> tensor<1x100x4xbf16> {
    // CHECK-LABEL: func.func @linear_with_gelu
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: -> tensor<100x4xbf16
    // CHECK: "ttnn.add"
    // CHECK-SAME: -> tensor<1x100x4xbf16
    // CHECK: "ttnn.gelu"
    // CHECK-SAME: -> tensor<1x100x4xbf16
    // CHECK-NOT: "ttnn.linear"
    // CHECK: return
    %result = "ttnn.linear"(%arg0, %arg1, %arg2) <{activation = "gelu", transpose_a = false, transpose_b = true}> : (tensor<100x384xbf16>, tensor<4x384xbf16>, tensor<1x100x4xbf16>) -> tensor<1x100x4xbf16>
    return %result : tensor<1x100x4xbf16>
  }
  func.func @linear_with_tanh(%arg0: tensor<100x384xbf16>, %arg1: tensor<4x384xbf16>, %arg2: tensor<1x100x4xbf16>) -> tensor<1x100x4xbf16> {
    // CHECK-LABEL: func.func @linear_with_tanh
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: -> tensor<100x4xbf16
    // CHECK: "ttnn.add"
    // CHECK-SAME: -> tensor<1x100x4xbf16
    // CHECK: "ttnn.tanh"
    // CHECK-SAME: -> tensor<1x100x4xbf16
    // CHECK-NOT: "ttnn.linear"
    // CHECK: return
    %result = "ttnn.linear"(%arg0, %arg1, %arg2) <{activation = "tanh", transpose_a = false, transpose_b = true}> : (tensor<100x384xbf16>, tensor<4x384xbf16>, tensor<1x100x4xbf16>) -> tensor<1x100x4xbf16>
    return %result : tensor<1x100x4xbf16>
  }
  func.func @linear_with_silu(%arg0: tensor<100x384xbf16>, %arg1: tensor<4x384xbf16>, %arg2: tensor<1x100x4xbf16>) -> tensor<1x100x4xbf16> {
    // CHECK-LABEL: func.func @linear_with_silu
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: -> tensor<100x4xbf16
    // CHECK: "ttnn.add"
    // CHECK-SAME: -> tensor<1x100x4xbf16
    // CHECK: "ttnn.silu"
    // CHECK-SAME: -> tensor<1x100x4xbf16
    // CHECK-NOT: "ttnn.linear"
    // CHECK: return
    %result = "ttnn.linear"(%arg0, %arg1, %arg2) <{activation = "silu", transpose_a = false, transpose_b = true}> : (tensor<100x384xbf16>, tensor<4x384xbf16>, tensor<1x100x4xbf16>) -> tensor<1x100x4xbf16>
    return %result : tensor<1x100x4xbf16>
  }
  func.func @linear_with_batched_bias(%arg0: tensor<2x33x1024xf32>, %arg1: tensor<1024x1024xf32>, %arg2: tensor<2x1x1xf32>) -> tensor<2x33x1024xf32>{
    // CHECK-LABEL: func.func @linear_with_batched_bias
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: -> tensor<2x33x1024xf32
    // CHECK: "ttnn.add"
    // CHECK-SAME: -> tensor<2x33x1024xf32
    %result = "ttnn.linear"(%arg0, %arg1, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<2x33x1024xf32>, tensor<1024x1024xf32>, tensor<2x1x1xf32>) -> tensor<2x33x1024xf32>
    return %result : tensor<2x33x1024xf32>
  }
  func.func @linear_with_tile_height_bias_decomposed(%arg0: tensor<32x1024xf32>, %arg1: tensor<1024x1024xf32>, %arg2: tensor<1x32x1024xf32>) -> tensor<32x1024xf32>{
    // CHECK-LABEL: func.func @linear_with_tile_height_bias_decomposed
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: -> tensor<32x1024xf32
    // CHECK: "ttnn.add"
    // CHECK-SAME: -> tensor<1x32x1024xf32
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: -> tensor<32x1024xf32
    // CHECK-NOT: "ttnn.linear"
    // CHECK: return
    %result = "ttnn.linear"(%arg0, %arg1, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<32x1024xf32>, tensor<1024x1024xf32>, tensor<1x32x1024xf32>) -> tensor<32x1024xf32>
    return %result : tensor<32x1024xf32>
  }

  func.func @linear_with_multirow_bias_under_tile_height(%arg0: tensor<17x1024xf32>, %arg1: tensor<1024x1024xf32>, %arg2: tensor<1x17x1024xf32>) -> tensor<17x1024xf32>{
    // CHECK-LABEL: func.func @linear_with_multirow_bias_under_tile_height
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: -> tensor<17x1024xf32
    // CHECK: "ttnn.add"
    // CHECK-SAME: -> tensor<1x17x1024xf32
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: -> tensor<17x1024xf32
    // CHECK-NOT: "ttnn.linear"
    // CHECK: return
    %result = "ttnn.linear"(%arg0, %arg1, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<17x1024xf32>, tensor<1024x1024xf32>, tensor<1x17x1024xf32>) -> tensor<17x1024xf32>
    return %result : tensor<17x1024xf32>
  }

  // Verify that ttcore.weight_dtype discardable attribute is preserved on the
  // matmul op after LinearOp decomposition into matmul + add.
  func.func @linear_decompose_preserves_weight_dtype(%arg0: tensor<2x33x1024xf32>, %arg1: tensor<2x1024x1024xf32>, %arg2: tensor<2x33x1024xf32>) -> tensor<2x33x1024xf32>{
    // CHECK-LABEL: func.func @linear_decompose_preserves_weight_dtype
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: ttcore.weight_dtype = "bfp_bf8"
    // CHECK: "ttnn.add"
    // CHECK-NOT: "ttnn.linear"
    // CHECK: return
    %result = "ttnn.linear"(%arg0, %arg1, %arg2) <{transpose_a = false, transpose_b = false}> {ttcore.weight_dtype = "bfp_bf8"} : (tensor<2x33x1024xf32>, tensor<2x1024x1024xf32>, tensor<2x33x1024xf32>) -> tensor<2x33x1024xf32>
    return %result : tensor<2x33x1024xf32>
  }

  func.func @linear_with_feature_broadcast_bias(%arg0: tensor<32x1024xf32>, %arg1: tensor<1024x1024xf32>, %arg2: tensor<1xf32>) -> tensor<32x1024xf32>{
    // CHECK-LABEL: func.func @linear_with_feature_broadcast_bias
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: -> tensor<32x1024xf32
    // CHECK: "ttnn.add"
    // CHECK-SAME: -> tensor<32x1024xf32
    // CHECK-NOT: "ttnn.reshape"
    // CHECK: return
    %result = "ttnn.linear"(%arg0, %arg1, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<32x1024xf32>, tensor<1024x1024xf32>, tensor<1xf32>) -> tensor<32x1024xf32>
    return %result : tensor<32x1024xf32>
  }

}
