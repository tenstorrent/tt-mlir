// RUN: ttmlir-opt --ttir-strided-conv-space-to-depth-opt -o %t %s
// RUN: FileCheck %s --input-file=%t

// The EVO50 stem: 7x7 stride-2 conv on a 3-channel NCHW image.
// Expect pixel_unshuffle(2) -> permute{0,2,3,1} -> conv2d stride 1, kernel 4x4,
// padding (1,2,1,2) from the original (2,2,3,3), weight repacked to [16,12,4,4].
// CHECK-LABEL: func.func @evo50_stem
func.func @evo50_stem(%arg0: tensor<1x3x384x1664xbf16>, %w: tensor<16x3x7x7xbf16>, %b: tensor<1x1x1x16xbf16>) -> tensor<1x192x832x16xbf16> {
  // Attributes print alphabetically: channel_order before downscale_factor.
  // CHECK: "ttir.pixel_unshuffle"
  // CHECK-SAME: channel_major
  // CHECK-SAME: downscale_factor = 2
  // CHECK-SAME: -> tensor<1x12x192x832xbf16>
  // CHECK: "ttir.permute"
  // CHECK-SAME: permutation = array<i64: 0, 2, 3, 1>
  // CHECK-SAME: -> tensor<1x192x832x12xbf16>
  // The weight repack chain ends at the packed [OC, C*4, kH/2, kW/2] weight.
  // CHECK: -> tensor<16x12x4x4xbf16>
  // CHECK: "ttir.conv2d"
  // CHECK-SAME: dilation = array<i32: 1, 1>
  // CHECK-SAME: padding = array<i32: 1, 1, 2, 2>
  // CHECK-SAME: stride = array<i32: 1, 1>
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x3x384x1664xbf16>) -> tensor<1x384x1664x3xbf16>
  %1 = "ttir.conv2d"(%0, %w, %b) <{stride = array<i32: 2, 2>, padding = array<i32: 2, 2, 3, 3>, dilation = array<i32: 1, 1>, groups = 1 : i32}> {channel_last} : (tensor<1x384x1664x3xbf16>, tensor<16x3x7x7xbf16>, tensor<1x1x1x16xbf16>) -> tensor<1x192x832x16xbf16>
  return %1 : tensor<1x192x832x16xbf16>
}
