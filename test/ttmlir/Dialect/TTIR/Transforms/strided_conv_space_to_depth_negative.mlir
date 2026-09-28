// RUN: ttmlir-opt --ttir-strided-conv-space-to-depth-opt -o %t %s
// RUN: FileCheck %s --input-file=%t

// Each case below must be left untouched: no pixel_unshuffle anywhere.
// CHECK-NOT: ttir.pixel_unshuffle

// stride 1 is not a strided stem conv.
func.func @not_stride2(%arg0: tensor<1x3x384x1664xbf16>, %w: tensor<16x3x7x7xbf16>) -> tensor<1x384x1664x16xbf16> {
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x3x384x1664xbf16>) -> tensor<1x384x1664x3xbf16>
  %1 = "ttir.conv2d"(%0, %w) <{stride = array<i32: 1, 1>, padding = array<i32: 3, 3, 3, 3>, dilation = array<i32: 1, 1>, groups = 1 : i32}> {channel_last} : (tensor<1x384x1664x3xbf16>, tensor<16x3x7x7xbf16>) -> tensor<1x384x1664x16xbf16>
  return %1 : tensor<1x384x1664x16xbf16>
}

// Odd top padding has no integer space-to-depth phase.
func.func @odd_top_pad(%arg0: tensor<1x3x384x1664xbf16>, %w: tensor<16x3x7x7xbf16>) -> tensor<1x192x832x16xbf16> {
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x3x384x1664xbf16>) -> tensor<1x384x1664x3xbf16>
  %1 = "ttir.conv2d"(%0, %w) <{stride = array<i32: 2, 2>, padding = array<i32: 3, 2, 2, 3>, dilation = array<i32: 1, 1>, groups = 1 : i32}> {channel_last} : (tensor<1x384x1664x3xbf16>, tensor<16x3x7x7xbf16>) -> tensor<1x192x832x16xbf16>
  return %1 : tensor<1x192x832x16xbf16>
}

// groups != 1.
func.func @grouped(%arg0: tensor<1x4x384x1664xbf16>, %w: tensor<16x2x7x7xbf16>) -> tensor<1x192x832x16xbf16> {
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x4x384x1664xbf16>) -> tensor<1x384x1664x4xbf16>
  %1 = "ttir.conv2d"(%0, %w) <{stride = array<i32: 2, 2>, padding = array<i32: 2, 2, 3, 3>, dilation = array<i32: 1, 1>, groups = 2 : i32}> {channel_last} : (tensor<1x384x1664x4xbf16>, tensor<16x2x7x7xbf16>) -> tensor<1x192x832x16xbf16>
  return %1 : tensor<1x192x832x16xbf16>
}

// C*4 = 64 > 32: packed channels would not fit a tile column.
func.func @too_many_channels(%arg0: tensor<1x16x384x1664xbf16>, %w: tensor<16x16x7x7xbf16>) -> tensor<1x192x832x16xbf16> {
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x16x384x1664xbf16>) -> tensor<1x384x1664x16xbf16>
  %1 = "ttir.conv2d"(%0, %w) <{stride = array<i32: 2, 2>, padding = array<i32: 2, 2, 3, 3>, dilation = array<i32: 1, 1>, groups = 1 : i32}> {channel_last} : (tensor<1x384x1664x16xbf16>, tensor<16x16x7x7xbf16>) -> tensor<1x192x832x16xbf16>
  return %1 : tensor<1x192x832x16xbf16>
}

// No channel_last: not the NHWC stem form.
func.func @no_channel_last(%arg0: tensor<1x3x384x1664xbf16>, %w: tensor<16x3x7x7xbf16>) -> tensor<1x192x832x16xbf16> {
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x3x384x1664xbf16>) -> tensor<1x384x1664x3xbf16>
  %1 = "ttir.conv2d"(%0, %w) <{stride = array<i32: 2, 2>, padding = array<i32: 2, 2, 3, 3>, dilation = array<i32: 1, 1>, groups = 1 : i32}> : (tensor<1x384x1664x3xbf16>, tensor<16x3x7x7xbf16>) -> tensor<1x192x832x16xbf16>
  return %1 : tensor<1x192x832x16xbf16>
}
