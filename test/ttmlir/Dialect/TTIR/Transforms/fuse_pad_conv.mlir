// RUN: ttmlir-opt --ttir-fusing %s | FileCheck %s

// Positive case: zero-fill `ttir.pad` on H/W with a single user is absorbed
// into the conv2d's `padding` attribute. The pad op is removed by DCE and
// the resulting IR has only `ttir.conv2d` with combined padding
// [pT+lowH, pL+lowW, pB+highH, pR+highW] = [1, 1, 1, 1].
// CHECK-LABEL: func.func @fuse_pad_into_conv2d
func.func @fuse_pad_into_conv2d(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>) -> tensor<1x32x32x64xbf16> {
  // CHECK-NOT: "ttir.pad"
  // CHECK: %[[CONV:.*]] = "ttir.conv2d"(%arg0, %arg1)
  // CHECK-SAME: padding = array<i32: 1, 1, 1, 1>
  // CHECK-NOT: "ttir.pad"
  // CHECK: return %[[CONV]]
  %0 = "ttir.pad"(%arg0) <{padding = array<i32: 0, 0, 1, 1, 1, 1, 0, 0>, value = 0.0 : f32}> : (tensor<1x32x32x64xbf16>) -> tensor<1x34x34x64xbf16>
  %1 = "ttir.conv2d"(%0, %arg1)
          <{
            stride = 1: i32,
            padding = 0: i32,
            dilation = 1: i32,
            groups = 1: i32
          }> : (tensor<1x34x34x64xbf16>, tensor<64x64x3x3xbf16>) -> tensor<1x32x32x64xbf16>
  return %1 : tensor<1x32x32x64xbf16>
}

// Negative case: pad `value` is non-zero (1.0) — cannot be absorbed because
// conv padding is always zero-fill. The pad op must remain.
// CHECK-LABEL: func.func @no_fuse_nonzero_pad_value
func.func @no_fuse_nonzero_pad_value(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>) -> tensor<1x32x32x64xbf16> {
  // CHECK: "ttir.pad"
  // CHECK: "ttir.conv2d"
  %0 = "ttir.pad"(%arg0) <{padding = array<i32: 0, 0, 1, 1, 1, 1, 0, 0>, value = 1.0 : f32}> : (tensor<1x32x32x64xbf16>) -> tensor<1x34x34x64xbf16>
  %1 = "ttir.conv2d"(%0, %arg1)
          <{
            stride = 1: i32,
            padding = 0: i32,
            dilation = 1: i32,
            groups = 1: i32
          }> : (tensor<1x34x34x64xbf16>, tensor<64x64x3x3xbf16>) -> tensor<1x32x32x64xbf16>
  return %1 : tensor<1x32x32x64xbf16>
}

// Negative case: non-zero pad on the batch (N) dimension cannot be absorbed
// into conv padding. The pad op must remain.
// CHECK-LABEL: func.func @no_fuse_pad_on_batch_dim
func.func @no_fuse_pad_on_batch_dim(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>) -> tensor<2x32x32x64xbf16> {
  // CHECK: "ttir.pad"
  // CHECK: "ttir.conv2d"
  %0 = "ttir.pad"(%arg0) <{padding = array<i32: 1, 0, 1, 1, 1, 1, 0, 0>, value = 0.0 : f32}> : (tensor<1x32x32x64xbf16>) -> tensor<2x34x34x64xbf16>
  %1 = "ttir.conv2d"(%0, %arg1)
          <{
            stride = 1: i32,
            padding = 0: i32,
            dilation = 1: i32,
            groups = 1: i32
          }> : (tensor<2x34x34x64xbf16>, tensor<64x64x3x3xbf16>) -> tensor<2x32x32x64xbf16>
  return %1 : tensor<2x32x32x64xbf16>
}
