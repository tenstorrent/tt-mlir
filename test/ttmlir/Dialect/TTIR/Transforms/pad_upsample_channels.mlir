// RUN: ttmlir-opt --ttir-pad-upsample-channels -o %t %s
// RUN: FileCheck %s --input-file=%t

// An 80-channel bilinear upsample feeding a conv: X is zero padded to 96
// channels once, the upsample runs on 96 channels and the conv gets 16 zero
// input channels in its weights.

module {
  // CHECK-LABEL: func.func @pad_into_conv
  // CHECK: %[[X:.*]] = "ttir.pad"(%arg0) <{padding = array<i32: 0, 0, 0, 0, 0, 0, 0, 16>, value = 0.000000e+00 : f32}> : (tensor<1x3x13x80xbf16>) -> tensor<1x3x13x96xbf16>
  // CHECK: %[[U:.*]] = "ttir.upsample2d"(%[[X]]) {{.*}} -> tensor<1x6x26x96xbf16>
  // CHECK: %[[W:.*]] = "ttir.pad"(%arg1) <{padding = array<i32: 0, 0, 0, 16, 0, 0, 0, 0>, value = 0.000000e+00 : f32}> : (tensor<80x80x3x3xbf16>) -> tensor<80x96x3x3xbf16>
  // CHECK: "ttir.conv2d"(%[[U]], %[[W]], %arg2)
  // CHECK-SAME: -> tensor<1x6x26x80xbf16>
  func.func @pad_into_conv(%x: tensor<1x3x13x80xbf16>,
                           %w: tensor<80x80x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                           %b: tensor<1x1x1x80xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<1x6x26x80xbf16> {
    %0 = "ttir.upsample2d"(%x) <{mode = "bilinear", scale_factor = array<i32: 2, 2>}> {channel_last = true} : (tensor<1x3x13x80xbf16>) -> tensor<1x6x26x80xbf16>
    %1 = "ttir.conv2d"(%0, %w, %b) <{batch_dim = 0 : i64, channel_dim = 3 : i64, dilation = array<i32: 1, 1>, groups = 1 : i32, height_dim = 1 : i64, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>, width_dim = 2 : i64}> {channel_last = true} : (tensor<1x6x26x80xbf16>, tensor<80x80x3x3xbf16>, tensor<1x1x1x80xbf16>) -> tensor<1x6x26x80xbf16>
    return %1 : tensor<1x6x26x80xbf16>
  }

  // Already a multiple of 32: unchanged.
  // CHECK-LABEL: func.func @aligned
  // CHECK-NOT: "ttir.pad"
  func.func @aligned(%x: tensor<1x3x13x64xbf16>,
                     %w: tensor<80x64x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<1x6x26x80xbf16> {
    %0 = "ttir.upsample2d"(%x) <{mode = "bilinear", scale_factor = array<i32: 2, 2>}> {channel_last = true} : (tensor<1x3x13x64xbf16>) -> tensor<1x6x26x64xbf16>
    %1 = "ttir.conv2d"(%0, %w) <{batch_dim = 0 : i64, channel_dim = 3 : i64, dilation = array<i32: 1, 1>, groups = 1 : i32, height_dim = 1 : i64, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>, width_dim = 2 : i64}> {channel_last = true} : (tensor<1x6x26x64xbf16>, tensor<80x64x3x3xbf16>) -> tensor<1x6x26x80xbf16>
    return %1 : tensor<1x6x26x80xbf16>
  }
}
