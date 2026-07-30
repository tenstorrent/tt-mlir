// RUN: ttmlir-opt --ttir-fusing %s | FileCheck %s

module {
  // F.normalize-style RMS norm over a non-last (channel) dim:
  //   (x / clamp_min(sqrt(sum(x^2, dim=1)), eps)) * sqrt(4)
  // The reduced dim (1) is permuted to last, normalized, then permuted back.
  // CHECK-LABEL: func.func @fuse_channel_rms
  func.func @fuse_channel_rms(%arg0: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
    // CHECK: "ttir.permute"
    // CHECK: "ttir.rms_norm"
    // CHECK-SAME: normalized_shape = array<i64: 4>
    // CHECK: "ttir.permute"
    // CHECK-NOT: "ttir.div"
    %two = "ttir.full"() <{shape = array<i32: 2, 4, 8>, fill_value = 2.0 : f32}> : () -> tensor<2x4x8xf32>
    %sq = "ttir.pow"(%arg0, %two) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    %sum = "ttir.sum"(%sq) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<2x4x8xf32>) -> tensor<2x1x8xf32>
    %half = "ttir.full"() <{shape = array<i32: 2, 1, 8>, fill_value = 0.5 : f32}> : () -> tensor<2x1x8xf32>
    %sqrt = "ttir.pow"(%sum, %half) : (tensor<2x1x8xf32>, tensor<2x1x8xf32>) -> tensor<2x1x8xf32>
    %min = "ttir.full"() <{shape = array<i32: 2, 1, 8>, fill_value = 1.000000e-12 : f32}> : () -> tensor<2x1x8xf32>
    %max = "ttir.full"() <{shape = array<i32: 2, 1, 8>, fill_value = 3.40282347E+38 : f32}> : () -> tensor<2x1x8xf32>
    %clamp = "ttir.clamp_tensor"(%sqrt, %min, %max) : (tensor<2x1x8xf32>, tensor<2x1x8xf32>, tensor<2x1x8xf32>) -> tensor<2x1x8xf32>
    %bcast = "ttir.broadcast"(%clamp) <{broadcast_dimensions = array<i64: 1, 4, 1>}> : (tensor<2x1x8xf32>) -> tensor<2x4x8xf32>
    %div = "ttir.div"(%arg0, %bcast) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    %scale = "ttir.full"() <{shape = array<i32: 2, 4, 8>, fill_value = 2.0 : f32}> : () -> tensor<2x4x8xf32>
    %out = "ttir.multiply"(%div, %scale) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    return %out : tensor<2x4x8xf32>
  }

  // Same as above but with a trailing per-channel gamma multiply, which is
  // folded into the rms_norm weight operand.
  // CHECK-LABEL: func.func @fuse_channel_rms_with_gamma
  func.func @fuse_channel_rms_with_gamma(%arg0: tensor<2x4x8xf32>, %g: tensor<4xf32>) -> tensor<2x4x8xf32> {
    // CHECK: "ttir.permute"
    // CHECK: "ttir.rms_norm"(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}})
    // CHECK-SAME: normalized_shape = array<i64: 4>
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 0>
    // CHECK: "ttir.permute"
    // CHECK-NOT: "ttir.div"
    %two = "ttir.full"() <{shape = array<i32: 2, 4, 8>, fill_value = 2.0 : f32}> : () -> tensor<2x4x8xf32>
    %sq = "ttir.pow"(%arg0, %two) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    %sum = "ttir.sum"(%sq) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<2x4x8xf32>) -> tensor<2x1x8xf32>
    %half = "ttir.full"() <{shape = array<i32: 2, 1, 8>, fill_value = 0.5 : f32}> : () -> tensor<2x1x8xf32>
    %sqrt = "ttir.pow"(%sum, %half) : (tensor<2x1x8xf32>, tensor<2x1x8xf32>) -> tensor<2x1x8xf32>
    %min = "ttir.full"() <{shape = array<i32: 2, 1, 8>, fill_value = 1.000000e-12 : f32}> : () -> tensor<2x1x8xf32>
    %max = "ttir.full"() <{shape = array<i32: 2, 1, 8>, fill_value = 3.40282347E+38 : f32}> : () -> tensor<2x1x8xf32>
    %clamp = "ttir.clamp_tensor"(%sqrt, %min, %max) : (tensor<2x1x8xf32>, tensor<2x1x8xf32>, tensor<2x1x8xf32>) -> tensor<2x1x8xf32>
    %bcast = "ttir.broadcast"(%clamp) <{broadcast_dimensions = array<i64: 1, 4, 1>}> : (tensor<2x1x8xf32>) -> tensor<2x4x8xf32>
    %div = "ttir.div"(%arg0, %bcast) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    %scale = "ttir.full"() <{shape = array<i32: 2, 4, 8>, fill_value = 2.0 : f32}> : () -> tensor<2x4x8xf32>
    %scaled = "ttir.multiply"(%div, %scale) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    %gr = "ttir.reshape"(%g) <{shape = [1 : i32, 4 : i32, 1 : i32]}> : (tensor<4xf32>) -> tensor<1x4x1xf32>
    %gb = "ttir.broadcast"(%gr) <{broadcast_dimensions = array<i64: 2, 1, 8>}> : (tensor<1x4x1xf32>) -> tensor<2x4x8xf32>
    %out = "ttir.multiply"(%scaled, %gb) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    return %out : tensor<2x4x8xf32>
  }

  // Same decomposition but over the last dim (with keep_dim=false + reshape).
  // No permutes are needed. The scale is sqrt(8).
  // CHECK-LABEL: func.func @fuse_last_dim_rms
  func.func @fuse_last_dim_rms(%arg0: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
    // CHECK: "ttir.rms_norm"
    // CHECK-SAME: normalized_shape = array<i64: 8>
    // CHECK-NOT: "ttir.permute"
    // CHECK-NOT: "ttir.div"
    %two = "ttir.full"() <{shape = array<i32: 2, 4, 8>, fill_value = 2.0 : f32}> : () -> tensor<2x4x8xf32>
    %sq = "ttir.pow"(%arg0, %two) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    %sum = "ttir.sum"(%sq) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<2x4x8xf32>) -> tensor<2x4xf32>
    %rs = "ttir.reshape"(%sum) <{shape = [2 : i32, 4 : i32, 1 : i32]}> : (tensor<2x4xf32>) -> tensor<2x4x1xf32>
    %half = "ttir.full"() <{shape = array<i32: 2, 4, 1>, fill_value = 0.5 : f32}> : () -> tensor<2x4x1xf32>
    %sqrt = "ttir.pow"(%rs, %half) : (tensor<2x4x1xf32>, tensor<2x4x1xf32>) -> tensor<2x4x1xf32>
    %min = "ttir.full"() <{shape = array<i32: 2, 4, 1>, fill_value = 1.000000e-12 : f32}> : () -> tensor<2x4x1xf32>
    %max = "ttir.full"() <{shape = array<i32: 2, 4, 1>, fill_value = 3.40282347E+38 : f32}> : () -> tensor<2x4x1xf32>
    %clamp = "ttir.clamp_tensor"(%sqrt, %min, %max) : (tensor<2x4x1xf32>, tensor<2x4x1xf32>, tensor<2x4x1xf32>) -> tensor<2x4x1xf32>
    %bcast = "ttir.broadcast"(%clamp) <{broadcast_dimensions = array<i64: 1, 1, 8>}> : (tensor<2x4x1xf32>) -> tensor<2x4x8xf32>
    %div = "ttir.div"(%arg0, %bcast) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    %scale = "ttir.full"() <{shape = array<i32: 2, 4, 8>, fill_value = 2.828427 : f32}> : () -> tensor<2x4x8xf32>
    %out = "ttir.multiply"(%div, %scale) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    return %out : tensor<2x4x8xf32>
  }

  // A large clamp floor is a meaningful clamp, not an eps guard, so it must not
  // be reinterpreted as an rms_norm epsilon.
  // CHECK-LABEL: func.func @no_fuse_large_clamp
  func.func @no_fuse_large_clamp(%arg0: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
    // CHECK-NOT: "ttir.rms_norm"
    %two = "ttir.full"() <{shape = array<i32: 2, 4, 8>, fill_value = 2.0 : f32}> : () -> tensor<2x4x8xf32>
    %sq = "ttir.pow"(%arg0, %two) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    %sum = "ttir.sum"(%sq) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<2x4x8xf32>) -> tensor<2x1x8xf32>
    %half = "ttir.full"() <{shape = array<i32: 2, 1, 8>, fill_value = 0.5 : f32}> : () -> tensor<2x1x8xf32>
    %sqrt = "ttir.pow"(%sum, %half) : (tensor<2x1x8xf32>, tensor<2x1x8xf32>) -> tensor<2x1x8xf32>
    %min = "ttir.full"() <{shape = array<i32: 2, 1, 8>, fill_value = 5.000000e-01 : f32}> : () -> tensor<2x1x8xf32>
    %max = "ttir.full"() <{shape = array<i32: 2, 1, 8>, fill_value = 3.40282347E+38 : f32}> : () -> tensor<2x1x8xf32>
    %clamp = "ttir.clamp_tensor"(%sqrt, %min, %max) : (tensor<2x1x8xf32>, tensor<2x1x8xf32>, tensor<2x1x8xf32>) -> tensor<2x1x8xf32>
    %bcast = "ttir.broadcast"(%clamp) <{broadcast_dimensions = array<i64: 1, 4, 1>}> : (tensor<2x1x8xf32>) -> tensor<2x4x8xf32>
    %div = "ttir.div"(%arg0, %bcast) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    %scale = "ttir.full"() <{shape = array<i32: 2, 4, 8>, fill_value = 2.0 : f32}> : () -> tensor<2x4x8xf32>
    %out = "ttir.multiply"(%div, %scale) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    return %out : tensor<2x4x8xf32>
  }

  // Plain L2 normalization: the rescale factor is NOT sqrt(D) (here 3.0 while
  // sqrt(4) = 2.0), so this must remain unfused (it is not an RMS norm).
  // CHECK-LABEL: func.func @no_fuse_plain_l2
  func.func @no_fuse_plain_l2(%arg0: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
    // CHECK-NOT: "ttir.rms_norm"
    %two = "ttir.full"() <{shape = array<i32: 2, 4, 8>, fill_value = 2.0 : f32}> : () -> tensor<2x4x8xf32>
    %sq = "ttir.pow"(%arg0, %two) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    %sum = "ttir.sum"(%sq) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<2x4x8xf32>) -> tensor<2x1x8xf32>
    %half = "ttir.full"() <{shape = array<i32: 2, 1, 8>, fill_value = 0.5 : f32}> : () -> tensor<2x1x8xf32>
    %sqrt = "ttir.pow"(%sum, %half) : (tensor<2x1x8xf32>, tensor<2x1x8xf32>) -> tensor<2x1x8xf32>
    %min = "ttir.full"() <{shape = array<i32: 2, 1, 8>, fill_value = 1.000000e-12 : f32}> : () -> tensor<2x1x8xf32>
    %max = "ttir.full"() <{shape = array<i32: 2, 1, 8>, fill_value = 3.40282347E+38 : f32}> : () -> tensor<2x1x8xf32>
    %clamp = "ttir.clamp_tensor"(%sqrt, %min, %max) : (tensor<2x1x8xf32>, tensor<2x1x8xf32>, tensor<2x1x8xf32>) -> tensor<2x1x8xf32>
    %bcast = "ttir.broadcast"(%clamp) <{broadcast_dimensions = array<i64: 1, 4, 1>}> : (tensor<2x1x8xf32>) -> tensor<2x4x8xf32>
    %div = "ttir.div"(%arg0, %bcast) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    %scale = "ttir.full"() <{shape = array<i32: 2, 4, 8>, fill_value = 3.0 : f32}> : () -> tensor<2x4x8xf32>
    %out = "ttir.multiply"(%div, %scale) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    return %out : tensor<2x4x8xf32>
  }
}
