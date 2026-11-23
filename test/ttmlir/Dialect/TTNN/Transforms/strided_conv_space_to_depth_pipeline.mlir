// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-strided-conv-space-to-depth-opt=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// End to end with the optimizer off: the stem conv is not height-sharded, so
// TTNNPixelUnshuffleChannelsLast cannot use its grid and height-shards the
// channels_last pixel_unshuffle on the worker grid instead (1024 pixels on 64
// cores, shard [16, 16]), followed by the layout conversion its users expect.

// CHECK-LABEL: func.func @stem_7x7_s2
func.func @stem_7x7_s2(%arg0: tensor<1x3x64x64xbf16>, %w: tensor<16x3x7x7xbf16>, %b: tensor<1x1x1x16xbf16>) -> tensor<1x32x32x16xbf16> {
  // CHECK: %[[PU:.*]] = "ttnn.pixel_unshuffle"
  // CHECK-SAME: channels_last = true
  // CHECK-SAME: memory_config = #ttnn.memory_config<#l1, <height_sharded>
  // CHECK-SAME: <16x16>
  // CHECK-SAME: padded_channels = 16 : ui32
  // The fallback converts the sharded result back to the layout its users
  // were built against; no NCHW->NHWC activation permute is left.
  // CHECK: "ttnn.to_memory_config"(%[[PU]])
  // CHECK-NOT: permutation = array<i64: 0, 2, 3, 1>
  // CHECK: "ttnn.conv2d"
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x3x64x64xbf16>) -> tensor<1x64x64x3xbf16>
  %1 = "ttir.conv2d"(%0, %w, %b) <{stride = array<i32: 2, 2>, padding = array<i32: 2, 2, 3, 3>, dilation = array<i32: 1, 1>, groups = 1 : i32}> {channel_last} : (tensor<1x64x64x3xbf16>, tensor<16x3x7x7xbf16>, tensor<1x1x1x16xbf16>) -> tensor<1x32x32x16xbf16>
  return %1 : tensor<1x32x32x16xbf16>
}
