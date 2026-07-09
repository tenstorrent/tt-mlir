// RUN: ttmlir-opt --ttir-fusing %s | FileCheck %s

module {
  // silu(rms_norm(x)) -> dit_rms_norm_unary_fused with activation = "silu".
  // CHECK-LABEL: func.func @fuse_silu
  func.func @fuse_silu(%arg0: tensor<32x512xf32>) -> tensor<32x512xf32> {
    // CHECK: "ttir.dit_rms_norm_unary_fused"(%arg0)
    // CHECK-SAME: activation = "silu"
    // CHECK-NOT: "ttir.silu"
    %0 = "ttir.rms_norm"(%arg0) <{normalized_shape = array<i64: 512>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<32x512xf32>) -> tensor<32x512xf32>
    %1 = "ttir.silu"(%0) : (tensor<32x512xf32>) -> tensor<32x512xf32>
    return %1 : tensor<32x512xf32>
  }

  // gelu(rms_norm(x, weight)) -> dit_rms_norm_unary_fused, preserving weight.
  // CHECK-LABEL: func.func @fuse_gelu_weight
  func.func @fuse_gelu_weight(%arg0: tensor<32x512xf32>, %arg1: tensor<512xf32>) -> tensor<32x512xf32> {
    // CHECK: "ttir.dit_rms_norm_unary_fused"(%arg0, %arg1)
    // CHECK-SAME: activation = "gelu"
    %0 = "ttir.rms_norm"(%arg0, %arg1) <{normalized_shape = array<i64: 512>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<32x512xf32>, tensor<512xf32>) -> tensor<32x512xf32>
    %1 = "ttir.gelu"(%0) : (tensor<32x512xf32>) -> tensor<32x512xf32>
    return %1 : tensor<32x512xf32>
  }

  // relu(rms_norm(x, weight, bias)) -> dit_rms_norm_unary_fused, preserving bias.
  // CHECK-LABEL: func.func @fuse_relu_bias
  func.func @fuse_relu_bias(%arg0: tensor<32x512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> tensor<32x512xf32> {
    // CHECK: "ttir.dit_rms_norm_unary_fused"(%arg0, %arg1, %arg2)
    // CHECK-SAME: activation = "relu"
    %0 = "ttir.rms_norm"(%arg0, %arg1, %arg2) <{normalized_shape = array<i64: 512>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<32x512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<32x512xf32>
    %1 = "ttir.relu"(%0) : (tensor<32x512xf32>) -> tensor<32x512xf32>
    return %1 : tensor<32x512xf32>
  }

  // The fusion looks through value-preserving shape ops (permute / reshape)
  // between the rms_norm and the activation, since the elementwise activation
  // commutes with them. The fused op replaces the rms_norm (keeping its shape)
  // and the shape ops keep operating on the fused result.
  // CHECK-LABEL: func.func @fuse_through_permute
  func.func @fuse_through_permute(%arg0: tensor<1x1x90x160x384xbf16>, %arg1: tensor<384xbf16>) -> tensor<1x384x1x90x160xbf16> {
    // CHECK: %[[F:.*]] = "ttir.dit_rms_norm_unary_fused"(%arg0, %arg1)
    // CHECK-SAME: activation = "silu"
    // CHECK-SAME: -> tensor<1x1x90x160x384xbf16>
    // CHECK: "ttir.permute"(%[[F]])
    // CHECK-NOT: "ttir.silu"
    %0 = "ttir.rms_norm"(%arg0, %arg1) <{epsilon = 9.99999974E-6 : f32, normalized_shape = array<i64: 384>, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x1x90x160x384xbf16>, tensor<384xbf16>) -> tensor<1x1x90x160x384xbf16>
    %1 = "ttir.permute"(%0) <{permutation = array<i64: 0, 4, 1, 2, 3>}> : (tensor<1x1x90x160x384xbf16>) -> tensor<1x384x1x90x160xbf16>
    %2 = "ttir.silu"(%1) : (tensor<1x384x1x90x160xbf16>) -> tensor<1x384x1x90x160xbf16>
    return %2 : tensor<1x384x1x90x160xbf16>
  }

  // Look-through also works across a chain (permute + reshape).
  // CHECK-LABEL: func.func @fuse_through_permute_reshape
  func.func @fuse_through_permute_reshape(%arg0: tensor<2x4x8xf32>, %arg1: tensor<8xf32>) -> tensor<2x32xf32> {
    // CHECK: %[[F:.*]] = "ttir.dit_rms_norm_unary_fused"(%arg0, %arg1)
    // CHECK-SAME: activation = "gelu"
    // CHECK-SAME: -> tensor<2x4x8xf32>
    // CHECK-NOT: "ttir.gelu"
    %0 = "ttir.rms_norm"(%arg0, %arg1) <{epsilon = 9.99999974E-6 : f32, normalized_shape = array<i64: 8>, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<2x4x8xf32>, tensor<8xf32>) -> tensor<2x4x8xf32>
    %1 = "ttir.permute"(%0) <{permutation = array<i64: 0, 2, 1>}> : (tensor<2x4x8xf32>) -> tensor<2x8x4xf32>
    %2 = "ttir.reshape"(%1) <{shape = [2 : i32, 32 : i32]}> : (tensor<2x8x4xf32>) -> tensor<2x32xf32>
    %3 = "ttir.gelu"(%2) : (tensor<2x32xf32>) -> tensor<2x32xf32>
    return %3 : tensor<2x32xf32>
  }

  // The shape op between rms_norm and activation has more than one use, so the
  // rewrite would change other consumers; the pair must remain unfused.
  // CHECK-LABEL: func.func @no_fuse_shape_op_multi_use
  func.func @no_fuse_shape_op_multi_use(%arg0: tensor<2x4x8xf32>, %arg1: tensor<8xf32>) -> (tensor<2x8x4xf32>, tensor<2x8x4xf32>) {
    // CHECK: "ttir.rms_norm"
    // CHECK: "ttir.silu"
    // CHECK-NOT: "ttir.dit_rms_norm_unary_fused"
    %0 = "ttir.rms_norm"(%arg0, %arg1) <{epsilon = 9.99999974E-6 : f32, normalized_shape = array<i64: 8>, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<2x4x8xf32>, tensor<8xf32>) -> tensor<2x4x8xf32>
    %1 = "ttir.permute"(%0) <{permutation = array<i64: 0, 2, 1>}> : (tensor<2x4x8xf32>) -> tensor<2x8x4xf32>
    %2 = "ttir.silu"(%1) : (tensor<2x8x4xf32>) -> tensor<2x8x4xf32>
    return %1, %2 : tensor<2x8x4xf32>, tensor<2x8x4xf32>
  }

  // Multi-dimensional normalized_shape is unsupported by the TTNN kernel, so
  // the rms_norm + silu pair must remain unfused.
  // CHECK-LABEL: func.func @no_fuse_multidim
  func.func @no_fuse_multidim(%arg0: tensor<32x512xf32>, %arg1: tensor<32x512xf32>) -> tensor<32x512xf32> {
    // CHECK: "ttir.rms_norm"
    // CHECK: "ttir.silu"
    // CHECK-NOT: "ttir.dit_rms_norm_unary_fused"
    %0 = "ttir.rms_norm"(%arg0, %arg1) <{normalized_shape = array<i64: 32, 512>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<32x512xf32>, tensor<32x512xf32>) -> tensor<32x512xf32>
    %1 = "ttir.silu"(%0) : (tensor<32x512xf32>) -> tensor<32x512xf32>
    return %1 : tensor<32x512xf32>
  }
}
