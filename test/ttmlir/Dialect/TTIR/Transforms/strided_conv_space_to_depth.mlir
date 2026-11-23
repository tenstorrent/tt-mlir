// RUN: ttmlir-opt --ttir-strided-conv-space-to-depth-opt -o %t %s
// RUN: FileCheck %s --input-file=%t

// 7x7 stride-2 stem conv on a 3-channel NCHW image. Expect a channels_last
// pixel_unshuffle(2) writing NHWC with the 12 packed channels zero-padded to
// 16, feeding a stride-1 4x4 conv directly (no permute), padding (1,2,1,2)
// from the original (2,2,3,3), and the weight repacked to [16,12,4,4] then
// zero-padded to 16 input channels.
// CHECK-LABEL: func.func @stem_7x7_s2
func.func @stem_7x7_s2(%arg0: tensor<1x3x384x1664xbf16>, %w: tensor<16x3x7x7xbf16>, %b: tensor<1x1x1x16xbf16>) -> tensor<1x192x832x16xbf16> {
  // Attributes print alphabetically.
  // CHECK: %[[PU:.*]] = "ttir.pixel_unshuffle"(%arg0)
  // CHECK-SAME: channel_major
  // CHECK-SAME: channels_last = true
  // CHECK-SAME: downscale_factor = 2
  // CHECK-SAME: padded_channels = 16 : ui32
  // CHECK-SAME: -> tensor<1x192x832x16xbf16>
  // CHECK-NOT: permutation = array<i64: 0, 2, 3, 1>
  // The weight repack chain ends at the packed [OC, C*4, kH/2, kW/2] weight,
  // zero-padded to the activation's 16 channels.
  // CHECK: %[[W:.*]] = "ttir.pad"({{.*}}padding = array<i32: 0, 0, 0, 4, 0, 0, 0, 0>
  // CHECK-SAME: (tensor<16x12x4x4xbf16>) -> tensor<16x16x4x4xbf16>
  // CHECK: "ttir.conv2d"(%[[PU]], %[[W]], %arg2)
  // CHECK-SAME: dilation = array<i32: 1, 1>
  // CHECK-SAME: padding = array<i32: 1, 1, 2, 2>
  // CHECK-SAME: stride = array<i32: 1, 1>
  // CHECK-SAME: (tensor<1x192x832x16xbf16>, tensor<16x16x4x4xbf16>, tensor<1x1x1x16xbf16>) -> tensor<1x192x832x16xbf16>
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x3x384x1664xbf16>) -> tensor<1x384x1664x3xbf16>
  %1 = "ttir.conv2d"(%0, %w, %b) <{stride = array<i32: 2, 2>, padding = array<i32: 2, 2, 3, 3>, dilation = array<i32: 1, 1>, groups = 1 : i32}> {channel_last} : (tensor<1x384x1664x3xbf16>, tensor<16x3x7x7xbf16>, tensor<1x1x1x16xbf16>) -> tensor<1x192x832x16xbf16>
  return %1 : tensor<1x192x832x16xbf16>
}

// 8 input channels pack to exactly one tile column: Cp = 32, no weight pad.
// CHECK-LABEL: func.func @stem_3x3_s2_c8
func.func @stem_3x3_s2_c8(%arg0: tensor<1x8x64x64xbf16>, %w: tensor<32x8x3x3xbf16>, %b: tensor<1x1x1x32xbf16>) -> tensor<1x32x32x32xbf16> {
  // CHECK: %[[PU:.*]] = "ttir.pixel_unshuffle"(%arg0)
  // CHECK-SAME: channels_last = true
  // CHECK-SAME: padded_channels = 32 : ui32
  // CHECK-SAME: -> tensor<1x32x32x32xbf16>
  // CHECK-NOT: "ttir.pad"{{.*}}-> tensor<32x32x2x2xbf16>
  // CHECK: "ttir.conv2d"(%[[PU]],
  // CHECK-SAME: stride = array<i32: 1, 1>
  // CHECK-SAME: tensor<32x32x2x2xbf16>
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x8x64x64xbf16>) -> tensor<1x64x64x8xbf16>
  %1 = "ttir.conv2d"(%0, %w, %b) <{stride = array<i32: 2, 2>, padding = array<i32: 0, 0, 1, 1>, dilation = array<i32: 1, 1>, groups = 1 : i32}> {channel_last} : (tensor<1x64x64x8xbf16>, tensor<32x8x3x3xbf16>, tensor<1x1x1x32xbf16>) -> tensor<1x32x32x32xbf16>
  return %1 : tensor<1x32x32x32xbf16>
}
