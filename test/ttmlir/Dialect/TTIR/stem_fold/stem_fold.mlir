// RUN: ttmlir-opt --ttir-stem-fold -o %t %s
// RUN: FileCheck %s --input-file=%t

// A linear image stem as a frontend emits it, 6D space-to-depth chains included
// (no TTIRPixelUnshuffleOpt in front): 1x1 mix conv 3->3 (+bias), direct branch
// space_to_depth(4) on channel 0, averaged branch [2,1]/stride-2 grouped average
// + space_to_depth(2) on channels 1..2, concat, 1x1 conv 24->64 (+bias), relu6.
// Input 1x3x1536x1536. After the fold: pixel_unshuffle(4, channels_last, 64) +
// one 1x1 conv whose weight/bias are built from the stem parameters, then the clamp.
module {
  func.func @stem(%arg0: tensor<1x3x1536x1536xbf16>,
                  %arg1: tensor<2x1x2x1xbf16>,
                  %arg7: tensor<3x3x1x1xbf16>, %arg8: tensor<1x1x1x3xbf16>,
                  %arg9: tensor<64x24x1x1xbf16>, %arg10: tensor<1x1x1x64xbf16>) -> tensor<1x384x384x64xbf16> {
    // CHECK-LABEL: func.func @stem
    // CHECK-NOT: "ttir.concat"(%{{.*}}) <{dim = -3
    // CHECK-NOT: "ttir.pixel_unshuffle"(%{{.*}}) <{channel_order = #ttir<pixel_unshuffle_channel_order spatial_major>
    // CHECK: %[[PU:[0-9]+]] = "ttir.pixel_unshuffle"(%arg0) <{channel_order = #ttir<pixel_unshuffle_channel_order channel_major>, channels_last = true, downscale_factor = 4 : ui32, padded_channels = 64 : ui32}> : (tensor<1x3x1536x1536xbf16>) -> tensor<1x384x384x64xbf16>
    // CHECK: %[[CONV:[0-9]+]] = "ttir.conv2d"(%[[PU]], %{{[0-9]+}}, %{{[0-9]+}})
    // CHECK-SAME: {channel_last = true}
    // CHECK-SAME: (tensor<1x384x384x64xbf16>, tensor<64x64x1x1xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x384x384x64xbf16>
    // CHECK: "ttir.clamp_scalar"(%[[CONV]])
    %0 = "ttir.transpose"(%arg0) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x3x1536x1536xbf16>) -> tensor<1x1536x3x1536xbf16>
    %1 = "ttir.transpose"(%0) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x1536x3x1536xbf16>) -> tensor<1x1536x1536x3xbf16>
    %2 = "ttir.conv2d"(%1, %arg7, %arg8) <{batch_dim = 0 : i64, channel_dim = 3 : i64, dilation = array<i32: 1, 1>, groups = 1 : i32, height_dim = 1 : i64, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>, width_dim = 2 : i64}> {channel_last = true} : (tensor<1x1536x1536x3xbf16>, tensor<3x3x1x1xbf16>, tensor<1x1x1x3xbf16>) -> tensor<1x1536x1536x3xbf16>
    %3 = "ttir.transpose"(%2) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x1536x1536x3xbf16>) -> tensor<1x1536x3x1536xbf16>
    %4 = "ttir.transpose"(%3) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x1536x3x1536xbf16>) -> tensor<1x3x1536x1536xbf16>
    %5 = "ttir.index"(%4) <{begin = 0 : i32, dim = 1 : i32, end = 1 : i32, step = 1 : i32}> : (tensor<1x3x1536x1536xbf16>) -> tensor<1x1x1536x1536xbf16>
    %6 = "ttir.reshape"(%5) <{shape = [1 : i32, 1 : i32, 384 : i32, 4 : i32, 384 : i32, 4 : i32]}> : (tensor<1x1x1536x1536xbf16>) -> tensor<1x1x384x4x384x4xbf16>
    %7 = "ttir.transpose"(%6) <{dim0 = -5 : si32, dim1 = -3 : si32}> : (tensor<1x1x384x4x384x4xbf16>) -> tensor<1x4x384x1x384x4xbf16>
    %8 = "ttir.transpose"(%7) <{dim0 = -4 : si32, dim1 = -1 : si32}> : (tensor<1x4x384x1x384x4xbf16>) -> tensor<1x4x4x1x384x384xbf16>
    %9 = "ttir.transpose"(%8) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x4x4x1x384x384xbf16>) -> tensor<1x4x4x1x384x384xbf16>
    %10 = "ttir.reshape"(%9) <{shape = [1 : i32, 16 : i32, 384 : i32, 384 : i32]}> : (tensor<1x4x4x1x384x384xbf16>) -> tensor<1x16x384x384xbf16>
    %11 = "ttir.index"(%4) <{begin = 1 : i32, dim = 1 : i32, end = 3 : i32, step = 1 : i32}> : (tensor<1x3x1536x1536xbf16>) -> tensor<1x2x1536x1536xbf16>
    %12 = "ttir.transpose"(%11) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x2x1536x1536xbf16>) -> tensor<1x1536x2x1536xbf16>
    %13 = "ttir.transpose"(%12) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x1536x2x1536xbf16>) -> tensor<1x1536x1536x2xbf16>
    %14 = "ttir.conv2d"(%13, %arg1) <{batch_dim = 0 : i64, channel_dim = 3 : i64, dilation = array<i32: 1, 1>, groups = 2 : i32, height_dim = 1 : i64, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 2, 2>, width_dim = 2 : i64}> {channel_last = true} : (tensor<1x1536x1536x2xbf16>, tensor<2x1x2x1xbf16>) -> tensor<1x768x768x2xbf16>
    %15 = "ttir.transpose"(%14) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x768x768x2xbf16>) -> tensor<1x768x2x768xbf16>
    %16 = "ttir.transpose"(%15) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x768x2x768xbf16>) -> tensor<1x2x768x768xbf16>
    %17 = "ttir.reshape"(%16) <{shape = [1 : i32, 2 : i32, 384 : i32, 2 : i32, 384 : i32, 2 : i32]}> : (tensor<1x2x768x768xbf16>) -> tensor<1x2x384x2x384x2xbf16>
    %18 = "ttir.transpose"(%17) <{dim0 = -5 : si32, dim1 = -3 : si32}> : (tensor<1x2x384x2x384x2xbf16>) -> tensor<1x2x384x2x384x2xbf16>
    %19 = "ttir.transpose"(%18) <{dim0 = -4 : si32, dim1 = -1 : si32}> : (tensor<1x2x384x2x384x2xbf16>) -> tensor<1x2x2x2x384x384xbf16>
    %20 = "ttir.transpose"(%19) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x2x2x2x384x384xbf16>) -> tensor<1x2x2x2x384x384xbf16>
    %21 = "ttir.reshape"(%20) <{shape = [1 : i32, 8 : i32, 384 : i32, 384 : i32]}> : (tensor<1x2x2x2x384x384xbf16>) -> tensor<1x8x384x384xbf16>
    %22 = "ttir.concat"(%10, %21) <{dim = -3 : si32}> : (tensor<1x16x384x384xbf16>, tensor<1x8x384x384xbf16>) -> tensor<1x24x384x384xbf16>
    %23 = "ttir.transpose"(%22) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x24x384x384xbf16>) -> tensor<1x384x24x384xbf16>
    %24 = "ttir.transpose"(%23) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x384x24x384xbf16>) -> tensor<1x384x384x24xbf16>
    %25 = "ttir.conv2d"(%24, %arg9, %arg10) <{batch_dim = 0 : i64, channel_dim = 3 : i64, dilation = array<i32: 1, 1>, groups = 1 : i32, height_dim = 1 : i64, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>, width_dim = 2 : i64}> {channel_last = true} : (tensor<1x384x384x24xbf16>, tensor<64x24x1x1xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x384x384x64xbf16>
    %26 = "ttir.clamp_scalar"(%25) <{max = 6.000000e+00 : f32, min = 0.000000e+00 : f32}> : (tensor<1x384x384x64xbf16>) -> tensor<1x384x384x64xbf16>
    return %26 : tensor<1x384x384x64xbf16>
  }

  // The same stem after canonicalization: transposes merged into permutes, index
  // canonicalized to slice_static, clamp(0,6) to relu6.
  func.func @stem_canonical(%arg0: tensor<1x3x1536x1536xbf16>,
                            %arg1: tensor<2x1x2x1xbf16>,
                            %arg7: tensor<3x3x1x1xbf16>, %arg8: tensor<1x1x1x3xbf16>,
                            %arg9: tensor<64x24x1x1xbf16>, %arg10: tensor<1x1x1x64xbf16>) -> tensor<1x384x384x64xbf16> {
    // CHECK-LABEL: func.func @stem_canonical
    // CHECK-NOT: "ttir.index"
    // CHECK: %[[PU:[0-9]+]] = "ttir.pixel_unshuffle"(%arg0) <{channel_order = #ttir<pixel_unshuffle_channel_order channel_major>, channels_last = true, downscale_factor = 4 : ui32, padded_channels = 64 : ui32}>
    // CHECK: %[[CONV:[0-9]+]] = "ttir.conv2d"(%[[PU]], %{{[0-9]+}}, %{{[0-9]+}})
    // CHECK: "ttir.relu6"(%[[CONV]])
    %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x3x1536x1536xbf16>) -> tensor<1x1536x1536x3xbf16>
    %1 = "ttir.conv2d"(%0, %arg7, %arg8) <{batch_dim = 0 : i64, channel_dim = 3 : i64, dilation = array<i32: 1, 1>, groups = 1 : i32, height_dim = 1 : i64, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>, width_dim = 2 : i64}> {channel_last = true} : (tensor<1x1536x1536x3xbf16>, tensor<3x3x1x1xbf16>, tensor<1x1x1x3xbf16>) -> tensor<1x1536x1536x3xbf16>
    %2 = "ttir.permute"(%1) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x1536x1536x3xbf16>) -> tensor<1x3x1536x1536xbf16>
    %3 = "ttir.slice_static"(%2) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 1536 : i32, 1536 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x3x1536x1536xbf16>) -> tensor<1x1x1536x1536xbf16>
    %4 = "ttir.pixel_unshuffle"(%3) <{channel_order = #ttir<pixel_unshuffle_channel_order spatial_major>, downscale_factor = 4 : ui32}> : (tensor<1x1x1536x1536xbf16>) -> tensor<1x16x384x384xbf16>
    %5 = "ttir.slice_static"(%2) <{begins = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 3 : i32, 1536 : i32, 1536 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x3x1536x1536xbf16>) -> tensor<1x2x1536x1536xbf16>
    %6 = "ttir.permute"(%5) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x2x1536x1536xbf16>) -> tensor<1x1536x1536x2xbf16>
    %7 = "ttir.conv2d"(%6, %arg1) <{batch_dim = 0 : i64, channel_dim = 3 : i64, dilation = array<i32: 1, 1>, groups = 2 : i32, height_dim = 1 : i64, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 2, 2>, width_dim = 2 : i64}> {channel_last = true} : (tensor<1x1536x1536x2xbf16>, tensor<2x1x2x1xbf16>) -> tensor<1x768x768x2xbf16>
    %8 = "ttir.permute"(%7) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x768x768x2xbf16>) -> tensor<1x2x768x768xbf16>
    %9 = "ttir.pixel_unshuffle"(%8) <{channel_order = #ttir<pixel_unshuffle_channel_order spatial_major>, downscale_factor = 2 : ui32}> : (tensor<1x2x768x768xbf16>) -> tensor<1x8x384x384xbf16>
    %10 = "ttir.concat"(%4, %9) <{dim = -3 : si32}> : (tensor<1x16x384x384xbf16>, tensor<1x8x384x384xbf16>) -> tensor<1x24x384x384xbf16>
    %11 = "ttir.permute"(%10) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x24x384x384xbf16>) -> tensor<1x384x384x24xbf16>
    %12 = "ttir.conv2d"(%11, %arg9, %arg10) <{batch_dim = 0 : i64, channel_dim = 3 : i64, dilation = array<i32: 1, 1>, groups = 1 : i32, height_dim = 1 : i64, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>, width_dim = 2 : i64}> {channel_last = true} : (tensor<1x384x384x24xbf16>, tensor<64x24x1x1xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x384x384x64xbf16>
    %13 = "ttir.relu6"(%12) : (tensor<1x384x384x64xbf16>) -> tensor<1x384x384x64xbf16>
    return %13 : tensor<1x384x384x64xbf16>
  }
}
