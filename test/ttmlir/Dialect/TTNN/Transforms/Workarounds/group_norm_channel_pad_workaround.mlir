// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-ttnn-decomposition-pass=false" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Tests for GroupNormChannelPadRewritePattern: when the GroupNorm channel
// dimension is not a multiple of the tile width (32), the workaround pads
// the input/weight/bias along C up to lcm(tile_width, channels_per_group),
// scales num_groups proportionally so channels_per_group is preserved, and
// slices the output back to the original channel size.

module {
  // Workaround SHOULD apply: C=16, num_groups=2 -> channels_per_group=8,
  // alignment=lcm(32, 8)=32, paddedC=32, paddedNumGroups=4.
  func.func @group_norm_unaligned_c16(
      %arg0: tensor<1x1x64x16xbf16>,
      %arg1: tensor<16xbf16>,
      %arg2: tensor<16xbf16>) -> tensor<1x1x64x16xbf16> {
    // CHECK-LABEL: func.func @group_norm_unaligned_c16
    // Input C is padded from 16 to 32.
    // CHECK: "ttnn.pad"(%arg0)
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0, 0, 0, 0, 16>
    // CHECK-SAME: -> tensor<1x1x64x32xbf16
    // Weight and bias are 1D-padded from 16 to 32.
    // CHECK: "ttnn.pad"
    // CHECK-SAME: padding = array<i32: 0, 16>
    // CHECK-SAME: tensor<16xbf16
    // CHECK-SAME: -> tensor<32xbf16
    // CHECK: "ttnn.pad"
    // CHECK-SAME: padding = array<i32: 0, 16>
    // CHECK-SAME: tensor<16xbf16
    // CHECK-SAME: -> tensor<32xbf16
    // GroupNorm runs on the padded shape with num_groups scaled 2 -> 4.
    // CHECK: "ttnn.group_norm"
    // CHECK-SAME: num_groups = 4
    // CHECK-SAME: -> tensor<1x1x64x32xbf16
    // Output is sliced back to the original C=16.
    // CHECK: "ttnn.slice_static"
    // CHECK-SAME: begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32]
    // CHECK-SAME: ends = [1 : i32, 1 : i32, 64 : i32, 16 : i32]
    // CHECK-SAME: step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]
    // CHECK-SAME: -> tensor<1x1x64x16xbf16
    %0 = "ttir.group_norm"(%arg0, %arg1, %arg2) <{num_groups = 2 : i64, epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 0, 1, 1>}> : (tensor<1x1x64x16xbf16>, tensor<16xbf16>, tensor<16xbf16>) -> tensor<1x1x64x16xbf16>
    return %0 : tensor<1x1x64x16xbf16>
  }

  // Workaround SHOULD apply with non-trivial pad: C=80, num_groups=5 ->
  // channels_per_group=16, alignment=lcm(32, 16)=32, paddedC=96,
  // paddedNumGroups=6.
  func.func @group_norm_unaligned_c80(
      %arg0: tensor<1x1x64x80xbf16>,
      %arg1: tensor<80xbf16>,
      %arg2: tensor<80xbf16>) -> tensor<1x1x64x80xbf16> {
    // CHECK-LABEL: func.func @group_norm_unaligned_c80
    // Input C is padded from 80 to 96 (next multiple of lcm(32, 16) = 32).
    // CHECK: "ttnn.pad"(%arg0)
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0, 0, 0, 0, 16>
    // CHECK-SAME: -> tensor<1x1x64x96xbf16
    // Weight and bias padded 80 -> 96.
    // CHECK: "ttnn.pad"
    // CHECK-SAME: padding = array<i32: 0, 16>
    // CHECK-SAME: tensor<80xbf16
    // CHECK-SAME: -> tensor<96xbf16
    // CHECK: "ttnn.pad"
    // CHECK-SAME: padding = array<i32: 0, 16>
    // CHECK-SAME: tensor<80xbf16
    // CHECK-SAME: -> tensor<96xbf16
    // GroupNorm with num_groups scaled 5 -> 6.
    // CHECK: "ttnn.group_norm"
    // CHECK-SAME: num_groups = 6
    // CHECK-SAME: -> tensor<1x1x64x96xbf16
    // Output sliced back to original C=80.
    // CHECK: "ttnn.slice_static"
    // CHECK-SAME: ends = [1 : i32, 1 : i32, 64 : i32, 80 : i32]
    // CHECK-SAME: -> tensor<1x1x64x80xbf16
    %0 = "ttir.group_norm"(%arg0, %arg1, %arg2) <{num_groups = 5 : i64, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 1, 1>}> : (tensor<1x1x64x80xbf16>, tensor<80xbf16>, tensor<80xbf16>) -> tensor<1x1x64x80xbf16>
    return %0 : tensor<1x1x64x80xbf16>
  }

  // Workaround should NOT apply: C=480 is already tile-aligned. No pad on
  // the input feeds into group_norm and no slice_static on the output.
  func.func @group_norm_aligned_no_workaround(
      %arg0: tensor<1x1x64x480xbf16>,
      %arg1: tensor<480xbf16>,
      %arg2: tensor<480xbf16>) -> tensor<1x1x64x480xbf16> {
    // CHECK-LABEL: func.func @group_norm_aligned_no_workaround
    // CHECK-NOT: "ttnn.pad"
    // CHECK-NOT: "ttnn.slice_static"
    // CHECK: "ttnn.group_norm"
    // CHECK-SAME: num_groups = 8
    // CHECK-SAME: -> tensor<1x1x64x480xbf16
    // CHECK-NOT: "ttnn.pad"
    // CHECK-NOT: "ttnn.slice_static"
    %0 = "ttir.group_norm"(%arg0, %arg1, %arg2) <{num_groups = 8 : i64, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 1, 1>}> : (tensor<1x1x64x480xbf16>, tensor<480xbf16>, tensor<480xbf16>) -> tensor<1x1x64x480xbf16>
    return %0 : tensor<1x1x64x480xbf16>
  }
}
