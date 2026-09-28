// RUN: ttmlir-opt --ttir-implicit-broadcast-fold --ttir-fusing="ttnn-enable-conv2d-with-multiply-pattern=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// A scalar scale (all dims 1) is uniform across output channels, so it can be
// folded into the convolution weights exactly like a per-channel scale.

module {
  // CHECK-LABEL: func.func @conv2d_scalar_scale
  func.func @conv2d_scalar_scale(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<1x1x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x30x30x64xbf16> {
    // CHECK: %[[WEIGHT_SCALED:.*]] = "ttir.multiply"
    // CHECK-SAME: (%arg1, %arg2
    // CHECK: %[[CONV:.*]] = "ttir.conv2d"
    // CHECK-SAME: (%arg0, %[[WEIGHT_SCALED]]
    // CHECK-NOT: "ttir.multiply"
    %1 = "ttir.conv2d"(%arg0, %arg1)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>) -> tensor<1x30x30x64xbf16>
    %2 = "ttir.multiply"(%1, %arg2) : (tensor<1x30x30x64xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x30x30x64xbf16>

    // CHECK: return %[[CONV]]
    return %2: tensor<1x30x30x64xbf16>
  }

  // Same, with the scalar on the left-hand side of the multiply, and on a
  // conv_transpose2d without bias. This is the shape the BEV camera backbone
  // produces.
  // CHECK-LABEL: func.func @conv_transpose2d_scalar_scale_lhs
  func.func @conv_transpose2d_scalar_scale_lhs(%arg0: tensor<1x12x12x192xbf16>, %arg1: tensor<192x192x2x2xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<1x1x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x24x24x192xbf16> {
    // CHECK: %[[WEIGHT_SCALED:.*]] = "ttir.multiply"
    // CHECK-SAME: (%arg1, %arg2
    // CHECK: %[[CONV:.*]] = "ttir.conv_transpose2d"
    // CHECK-SAME: (%arg0, %[[WEIGHT_SCALED]]
    // CHECK-NOT: "ttir.multiply"
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1)
            <{
              batch_dim = 0 : i64,
              channel_dim = 3 : i64,
              dilation = array<i32: 1, 1>,
              groups = 1 : i32,
              height_dim = 1 : i64,
              output_padding = array<i32: 0, 0>,
              padding = array<i32: 0, 0>,
              stride = array<i32: 2, 2>,
              width_dim = 2 : i64
            }> {channel_last = true} : (tensor<1x12x12x192xbf16>, tensor<192x192x2x2xbf16>) -> tensor<1x24x24x192xbf16>
    %2 = "ttir.multiply"(%arg2, %1) : (tensor<1x1x1x1xbf16>, tensor<1x24x24x192xbf16>) -> tensor<1x24x24x192xbf16>

    // CHECK: return %[[CONV]]
    return %2: tensor<1x24x24x192xbf16>
  }

  // A scale with a non-1 size outside the output-feature dim is not uniform
  // across channels, so it must not be folded. (A scale whose channel dim is
  // neither 1 nor out_channels is not broadcast-compatible with the conv
  // result in the first place, so it cannot reach this pattern.)
  // CHECK-LABEL: func.func @conv2d_non_channel_scale_not_folded
  func.func @conv2d_non_channel_scale_not_folded(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<1x30x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x30x30x64xbf16> {
    // CHECK: "ttir.conv2d"
    // CHECK: "ttir.multiply"
    %1 = "ttir.conv2d"(%arg0, %arg1)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>) -> tensor<1x30x30x64xbf16>
    %2 = "ttir.multiply"(%1, %arg2) : (tensor<1x30x30x64xbf16>, tensor<1x30x1x1xbf16>) -> tensor<1x30x30x64xbf16>
    return %2: tensor<1x30x30x64xbf16>
  }

  // With bias the scale must land on both: (W x + b) * s == (W s) x + (b s).
  // CHECK-LABEL: func.func @conv2d_scalar_scale_with_bias
  func.func @conv2d_scalar_scale_with_bias(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<1x1x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg3: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x30x30x64xbf16> {
    // CHECK: %[[WEIGHT_SCALED:.*]] = "ttir.multiply"
    // CHECK-SAME: (%arg1, %arg2
    // CHECK: %[[BIAS_SCALED:.*]] = "ttir.multiply"
    // CHECK-SAME: (%arg3, %arg2
    // CHECK: %[[CONV:.*]] = "ttir.conv2d"
    // CHECK-SAME: (%arg0, %[[WEIGHT_SCALED]], %[[BIAS_SCALED]]
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg3) <{stride = 1: i32, padding = 0: i32, dilation = 1: i32, groups = 1: i32}> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16>
    %2 = "ttir.multiply"(%1, %arg2) : (tensor<1x30x30x64xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x30x30x64xbf16>
    // CHECK: return %[[CONV]]
    return %2: tensor<1x30x30x64xbf16>
  }

  // A scalar scale is exact for grouped and depthwise convolutions too.
  // CHECK-LABEL: func.func @conv2d_scalar_scale_depthwise
  func.func @conv2d_scalar_scale_depthwise(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x1x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<1x1x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x30x30x64xbf16> {
    // CHECK: %[[WEIGHT_SCALED:.*]] = "ttir.multiply"
    // CHECK-SAME: (%arg1, %arg2
    // CHECK: %[[CONV:.*]] = "ttir.conv2d"
    // CHECK-SAME: (%arg0, %[[WEIGHT_SCALED]]
    // CHECK-NOT: "ttir.multiply"
    %1 = "ttir.conv2d"(%arg0, %arg1) <{stride = 1: i32, padding = 0: i32, dilation = 1: i32, groups = 64: i32}> : (tensor<1x32x32x64xbf16>, tensor<64x1x3x3xbf16>) -> tensor<1x30x30x64xbf16>
    %2 = "ttir.multiply"(%1, %arg2) : (tensor<1x30x30x64xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x30x30x64xbf16>
    // CHECK: return %[[CONV]]
    return %2: tensor<1x30x30x64xbf16>
  }

  // A scalar that is a runtime input is not const-evalable, so it must stay.
  // CHECK-LABEL: func.func @conv2d_scalar_scale_not_constant
  func.func @conv2d_scalar_scale_not_constant(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<1x1x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<1x30x30x64xbf16> {
    // CHECK: "ttir.conv2d"
    // CHECK: "ttir.multiply"
    %1 = "ttir.conv2d"(%arg0, %arg1) <{stride = 1: i32, padding = 0: i32, dilation = 1: i32, groups = 1: i32}> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>) -> tensor<1x30x30x64xbf16>
    %2 = "ttir.multiply"(%1, %arg2) : (tensor<1x30x30x64xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x30x30x64xbf16>
    return %2: tensor<1x30x30x64xbf16>
  }

  // The conv result is used twice, so folding would change the other user.
  // CHECK-LABEL: func.func @conv2d_scalar_scale_two_uses
  func.func @conv2d_scalar_scale_two_uses(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<1x1x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x30x30x64xbf16> {
    // CHECK: "ttir.conv2d"
    // CHECK: "ttir.multiply"
    // CHECK: "ttir.add"
    %1 = "ttir.conv2d"(%arg0, %arg1) <{stride = 1: i32, padding = 0: i32, dilation = 1: i32, groups = 1: i32}> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>) -> tensor<1x30x30x64xbf16>
    %2 = "ttir.multiply"(%1, %arg2) : (tensor<1x30x30x64xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x30x30x64xbf16>
    %3 = "ttir.add"(%1, %2) : (tensor<1x30x30x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %3: tensor<1x30x30x64xbf16>
  }
}
