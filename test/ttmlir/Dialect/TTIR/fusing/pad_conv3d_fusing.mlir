// RUN: ttmlir-opt --ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

// Positive: symmetric, zero-valued pad on spatial dims fuses into conv3d.
// Layout is NCDHW (matches the StableHLO -> TTIR conv3d lowering).
module {
  // CHECK-LABEL: func.func @pad_conv3d_fuse
  func.func @pad_conv3d_fuse(%arg0: tensor<1x512x6x120x208xbf16>, %arg1: tensor<512x512x3x3x3xbf16>) -> tensor<1x512x4x120x208xbf16> {
    // CHECK-NOT: ttir.pad
    // CHECK: %[[CONV:.*]] = "ttir.conv3d"(%arg0, %arg1)
    // CHECK-SAME: padding = array<i32: 0, 1, 1>
    // CHECK-SAME: padding_mode = "zeros"
    // CHECK-SAME: stride = array<i32: 1, 1, 1>
    // CHECK-NEXT: return %[[CONV]]
    %0 = "ttir.pad"(%arg0) <{padding = array<i32: 0, 0, 0, 0, 0, 0, 1, 1, 1, 1>, value = 0.000000e+00 : f32}> : (tensor<1x512x6x120x208xbf16>) -> tensor<1x512x6x122x210xbf16>
    %1 = "ttir.conv3d"(%0, %arg1) <{batch_dim = 0 : i64, channel_dim = 1 : i64, depth_dim = 2 : i64, groups = 1 : i32, height_dim = 3 : i64, padding = array<i32: 0, 0, 0>, padding_mode = "zeros", stride = array<i32: 1, 1, 1>, width_dim = 4 : i64}> : (tensor<1x512x6x122x210xbf16>, tensor<512x512x3x3x3xbf16>) -> tensor<1x512x4x120x208xbf16>
    return %1 : tensor<1x512x4x120x208xbf16>
  }
}

// Positive: pad amounts add to existing conv3d padding.
module {
  // CHECK-LABEL: func.func @pad_conv3d_sum
  func.func @pad_conv3d_sum(%arg0: tensor<1x512x6x120x208xbf16>, %arg1: tensor<512x512x3x3x3xbf16>) -> tensor<1x512x6x122x210xbf16> {
    // CHECK-NOT: ttir.pad
    // CHECK: "ttir.conv3d"
    // CHECK-SAME: padding = array<i32: 1, 2, 2>
    %0 = "ttir.pad"(%arg0) <{padding = array<i32: 0, 0, 0, 0, 0, 0, 1, 1, 1, 1>, value = 0.000000e+00 : f32}> : (tensor<1x512x6x120x208xbf16>) -> tensor<1x512x6x122x210xbf16>
    %1 = "ttir.conv3d"(%0, %arg1) <{batch_dim = 0 : i64, channel_dim = 1 : i64, depth_dim = 2 : i64, groups = 1 : i32, height_dim = 3 : i64, padding = array<i32: 1, 1, 1>, padding_mode = "zeros", stride = array<i32: 1, 1, 1>, width_dim = 4 : i64}> : (tensor<1x512x6x122x210xbf16>, tensor<512x512x3x3x3xbf16>) -> tensor<1x512x6x122x210xbf16>
    return %1 : tensor<1x512x6x122x210xbf16>
  }
}

// Negative: asymmetric pad cannot be folded into conv3d (which requires symmetric padding).
module {
  // CHECK-LABEL: func.func @pad_conv3d_asymmetric
  func.func @pad_conv3d_asymmetric(%arg0: tensor<1x512x6x120x208xbf16>, %arg1: tensor<512x512x3x3x3xbf16>) -> tensor<1x512x4x119x207xbf16> {
    // CHECK: ttir.pad
    // CHECK: ttir.conv3d
    %0 = "ttir.pad"(%arg0) <{padding = array<i32: 0, 0, 0, 0, 0, 0, 1, 0, 1, 0>, value = 0.000000e+00 : f32}> : (tensor<1x512x6x120x208xbf16>) -> tensor<1x512x6x121x209xbf16>
    %1 = "ttir.conv3d"(%0, %arg1) <{batch_dim = 0 : i64, channel_dim = 1 : i64, depth_dim = 2 : i64, groups = 1 : i32, height_dim = 3 : i64, padding = array<i32: 0, 0, 0>, padding_mode = "zeros", stride = array<i32: 1, 1, 1>, width_dim = 4 : i64}> : (tensor<1x512x6x121x209xbf16>, tensor<512x512x3x3x3xbf16>) -> tensor<1x512x4x119x207xbf16>
    return %1 : tensor<1x512x4x119x207xbf16>
  }
}

// Negative: nonzero pad value is incompatible with conv3d "zeros" padding mode.
module {
  // CHECK-LABEL: func.func @pad_conv3d_nonzero_value
  func.func @pad_conv3d_nonzero_value(%arg0: tensor<1x512x6x120x208xbf16>, %arg1: tensor<512x512x3x3x3xbf16>) -> tensor<1x512x4x120x208xbf16> {
    // CHECK: ttir.pad
    // CHECK: ttir.conv3d
    %0 = "ttir.pad"(%arg0) <{padding = array<i32: 0, 0, 0, 0, 0, 0, 1, 1, 1, 1>, value = 1.000000e+00 : f32}> : (tensor<1x512x6x120x208xbf16>) -> tensor<1x512x6x122x210xbf16>
    %1 = "ttir.conv3d"(%0, %arg1) <{batch_dim = 0 : i64, channel_dim = 1 : i64, depth_dim = 2 : i64, groups = 1 : i32, height_dim = 3 : i64, padding = array<i32: 0, 0, 0>, padding_mode = "zeros", stride = array<i32: 1, 1, 1>, width_dim = 4 : i64}> : (tensor<1x512x6x122x210xbf16>, tensor<512x512x3x3x3xbf16>) -> tensor<1x512x4x120x208xbf16>
    return %1 : tensor<1x512x4x120x208xbf16>
  }
}

// Negative: padding the channel dim cannot be folded into conv3d.
module {
  // CHECK-LABEL: func.func @pad_conv3d_channel_pad
  func.func @pad_conv3d_channel_pad(%arg0: tensor<1x510x6x120x208xbf16>, %arg1: tensor<512x512x3x3x3xbf16>) -> tensor<1x512x4x120x208xbf16> {
    // CHECK: ttir.pad
    // CHECK: ttir.conv3d
    %0 = "ttir.pad"(%arg0) <{padding = array<i32: 0, 0, 1, 1, 0, 0, 1, 1, 1, 1>, value = 0.000000e+00 : f32}> : (tensor<1x510x6x120x208xbf16>) -> tensor<1x512x6x122x210xbf16>
    %1 = "ttir.conv3d"(%0, %arg1) <{batch_dim = 0 : i64, channel_dim = 1 : i64, depth_dim = 2 : i64, groups = 1 : i32, height_dim = 3 : i64, padding = array<i32: 0, 0, 0>, padding_mode = "zeros", stride = array<i32: 1, 1, 1>, width_dim = 4 : i64}> : (tensor<1x512x6x122x210xbf16>, tensor<512x512x3x3x3xbf16>) -> tensor<1x512x4x120x208xbf16>
    return %1 : tensor<1x512x4x120x208xbf16>
  }
}

// Negative: conv3d padding mode "replicate" cannot absorb a zero pad.
module {
  // CHECK-LABEL: func.func @pad_conv3d_replicate_mode
  func.func @pad_conv3d_replicate_mode(%arg0: tensor<1x512x6x120x208xbf16>, %arg1: tensor<512x512x3x3x3xbf16>) -> tensor<1x512x4x120x208xbf16> {
    // CHECK: ttir.pad
    // CHECK: ttir.conv3d
    %0 = "ttir.pad"(%arg0) <{padding = array<i32: 0, 0, 0, 0, 0, 0, 1, 1, 1, 1>, value = 0.000000e+00 : f32}> : (tensor<1x512x6x120x208xbf16>) -> tensor<1x512x6x122x210xbf16>
    %1 = "ttir.conv3d"(%0, %arg1) <{batch_dim = 0 : i64, channel_dim = 1 : i64, depth_dim = 2 : i64, groups = 1 : i32, height_dim = 3 : i64, padding = array<i32: 0, 0, 0>, padding_mode = "replicate", stride = array<i32: 1, 1, 1>, width_dim = 4 : i64}> : (tensor<1x512x6x122x210xbf16>, tensor<512x512x3x3x3xbf16>) -> tensor<1x512x4x120x208xbf16>
    return %1 : tensor<1x512x4x120x208xbf16>
  }
}

// Negative: pad with multiple uses cannot be fused (would still leave the pad alive).
module {
  // CHECK-LABEL: func.func @pad_conv3d_multiple_uses
  func.func @pad_conv3d_multiple_uses(%arg0: tensor<1x512x6x120x208xbf16>, %arg1: tensor<512x512x3x3x3xbf16>) -> (tensor<1x512x4x120x208xbf16>, tensor<1x512x6x122x210xbf16>) {
    // CHECK: ttir.pad
    // CHECK: ttir.conv3d
    %0 = "ttir.pad"(%arg0) <{padding = array<i32: 0, 0, 0, 0, 0, 0, 1, 1, 1, 1>, value = 0.000000e+00 : f32}> : (tensor<1x512x6x120x208xbf16>) -> tensor<1x512x6x122x210xbf16>
    %1 = "ttir.conv3d"(%0, %arg1) <{batch_dim = 0 : i64, channel_dim = 1 : i64, depth_dim = 2 : i64, groups = 1 : i32, height_dim = 3 : i64, padding = array<i32: 0, 0, 0>, padding_mode = "zeros", stride = array<i32: 1, 1, 1>, width_dim = 4 : i64}> : (tensor<1x512x6x122x210xbf16>, tensor<512x512x3x3x3xbf16>) -> tensor<1x512x4x120x208xbf16>
    return %1, %0 : tensor<1x512x4x120x208xbf16>, tensor<1x512x6x122x210xbf16>
  }
}
