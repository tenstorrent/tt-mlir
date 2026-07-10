// RUN: ttmlir-opt --split-input-file --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verifies the Conv3dOp canonicalizer: a 1x1x1 conv3d in NDHWC layout with
// unit stride, no padding and groups==1 is mathematically a matmul, so the
// canonicalizer rewrites it to ttir.matmul (no bias) / ttir.linear (bias)
// instead of lowering it as a conv3d. Anything failing the eligibility gate
// (isConv3dPointwiseLinearEligible) must stay a ttir.conv3d.

// -----

// Pointwise 1x1x1 conv3d, no bias -> ttir.matmul, conv3d eliminated.
// input  (N=1, D=8, H=16, W=16, Cin=64)
// weight (Cout=128, Cin=64, 1, 1, 1)  -> output (1, 8, 16, 16, 128)
func.func @pointwise_no_bias(%input: tensor<1x8x16x16x64xbf16>,
                             %weight: tensor<128x64x1x1x1xbf16>) -> tensor<1x8x16x16x128xbf16> {
  // CHECK-LABEL: @pointwise_no_bias
  // CHECK: "ttir.matmul"
  // CHECK-NOT: "ttir.linear"
  // CHECK-NOT: "ttir.conv3d"
  %0 = "ttir.conv3d"(%input, %weight)
          <{
            stride = array<i32: 1, 1, 1>,
            padding = array<i32: 0, 0, 0>,
            padding_mode = "zeros",
            groups = 1 : i32
          }> : (tensor<1x8x16x16x64xbf16>, tensor<128x64x1x1x1xbf16>) -> tensor<1x8x16x16x128xbf16>
  return %0 : tensor<1x8x16x16x128xbf16>
}

// -----

// Pointwise 1x1x1 conv3d, with bias -> ttir.linear, conv3d eliminated.
// bias (1, 1, 1, 1, Cout=128)
func.func @pointwise_with_bias(%input: tensor<1x8x16x16x64xbf16>,
                               %weight: tensor<128x64x1x1x1xbf16>,
                               %bias: tensor<1x1x1x1x128xbf16>) -> tensor<1x8x16x16x128xbf16> {
  // CHECK-LABEL: @pointwise_with_bias
  // CHECK: "ttir.linear"
  // CHECK-NOT: "ttir.conv3d"
  %0 = "ttir.conv3d"(%input, %weight, %bias)
          <{
            stride = array<i32: 1, 1, 1>,
            padding = array<i32: 0, 0, 0>,
            padding_mode = "zeros",
            groups = 1 : i32
          }> : (tensor<1x8x16x16x64xbf16>, tensor<128x64x1x1x1xbf16>, tensor<1x1x1x1x128xbf16>) -> tensor<1x8x16x16x128xbf16>
  return %0 : tensor<1x8x16x16x128xbf16>
}

// -----

// Multi-frame batch (N>1, D>1) must still fold: the rewrite collapses
// N*D*H*W into the matmul row dim, so this guards that reshape.
// input (N=2, D=4, H=8, W=8, Cin=64) -> output (2, 4, 8, 8, 128)
func.func @pointwise_multiframe(%input: tensor<2x4x8x8x64xbf16>,
                                %weight: tensor<128x64x1x1x1xbf16>) -> tensor<2x4x8x8x128xbf16> {
  // CHECK-LABEL: @pointwise_multiframe
  // CHECK: "ttir.matmul"
  // CHECK-NOT: "ttir.conv3d"
  %0 = "ttir.conv3d"(%input, %weight)
          <{
            stride = array<i32: 1, 1, 1>,
            padding = array<i32: 0, 0, 0>,
            padding_mode = "zeros",
            groups = 1 : i32
          }> : (tensor<2x4x8x8x64xbf16>, tensor<128x64x1x1x1xbf16>) -> tensor<2x4x8x8x128xbf16>
  return %0 : tensor<2x4x8x8x128xbf16>
}

// -----

// Negative: 3x3x3 kernel is a real convolution, must NOT be rewritten.
// output (1, 6, 14, 14, 128) from D_out=(8-3)+1, H/W_out=(16-3)+1.
func.func @negative_non_pointwise_kernel(%input: tensor<1x8x16x16x64xbf16>,
                                         %weight: tensor<128x64x3x3x3xbf16>) -> tensor<1x6x14x14x128xbf16> {
  // CHECK-LABEL: @negative_non_pointwise_kernel
  // CHECK: "ttir.conv3d"
  // CHECK-NOT: "ttir.matmul"
  // CHECK-NOT: "ttir.linear"
  %0 = "ttir.conv3d"(%input, %weight)
          <{
            stride = array<i32: 1, 1, 1>,
            padding = array<i32: 0, 0, 0>,
            padding_mode = "zeros",
            groups = 1 : i32
          }> : (tensor<1x8x16x16x64xbf16>, tensor<128x64x3x3x3xbf16>) -> tensor<1x6x14x14x128xbf16>
  return %0 : tensor<1x6x14x14x128xbf16>
}

// -----

// Negative: 1x1x1 kernel but non-unit stride is a strided gather, not a matmul.
// stride 2 -> D_out=(8-1)/2+1=4, H/W_out=(16-1)/2+1=8 -> output (1, 4, 8, 8, 128).
func.func @negative_strided(%input: tensor<1x8x16x16x64xbf16>,
                            %weight: tensor<128x64x1x1x1xbf16>) -> tensor<1x4x8x8x128xbf16> {
  // CHECK-LABEL: @negative_strided
  // CHECK: "ttir.conv3d"
  // CHECK-NOT: "ttir.matmul"
  %0 = "ttir.conv3d"(%input, %weight)
          <{
            stride = array<i32: 2, 2, 2>,
            padding = array<i32: 0, 0, 0>,
            padding_mode = "zeros",
            groups = 1 : i32
          }> : (tensor<1x8x16x16x64xbf16>, tensor<128x64x1x1x1xbf16>) -> tensor<1x4x8x8x128xbf16>
  return %0 : tensor<1x4x8x8x128xbf16>
}

// -----

// Negative: 1x1x1 kernel but non-zero padding, must NOT be rewritten.
// padding 1 -> D_out=(8+2-1)+1=10, H/W_out=(16+2-1)+1=18 -> output (1, 10, 18, 18, 128).
func.func @negative_padded(%input: tensor<1x8x16x16x64xbf16>,
                           %weight: tensor<128x64x1x1x1xbf16>) -> tensor<1x10x18x18x128xbf16> {
  // CHECK-LABEL: @negative_padded
  // CHECK: "ttir.conv3d"
  // CHECK-NOT: "ttir.matmul"
  %0 = "ttir.conv3d"(%input, %weight)
          <{
            stride = array<i32: 1, 1, 1>,
            padding = array<i32: 1, 1, 1>,
            padding_mode = "zeros",
            groups = 1 : i32
          }> : (tensor<1x8x16x16x64xbf16>, tensor<128x64x1x1x1xbf16>) -> tensor<1x10x18x18x128xbf16>
  return %0 : tensor<1x10x18x18x128xbf16>
}

// -----

// Negative: grouped conv (groups != 1), must NOT be rewritten.
// groups=2 -> weight C_in dim is Cin/groups = 32.
func.func @negative_grouped(%input: tensor<1x8x16x16x64xbf16>,
                            %weight: tensor<128x32x1x1x1xbf16>) -> tensor<1x8x16x16x128xbf16> {
  // CHECK-LABEL: @negative_grouped
  // CHECK: "ttir.conv3d"
  // CHECK-NOT: "ttir.matmul"
  %0 = "ttir.conv3d"(%input, %weight)
          <{
            stride = array<i32: 1, 1, 1>,
            padding = array<i32: 0, 0, 0>,
            padding_mode = "zeros",
            groups = 2 : i32
          }> : (tensor<1x8x16x16x64xbf16>, tensor<128x32x1x1x1xbf16>) -> tensor<1x8x16x16x128xbf16>
  return %0 : tensor<1x8x16x16x128xbf16>
}

// -----

// Negative: non-NDHWC (NCDHW) layout fails the isNDHWC() gate, so the
// canonicalizer leaves it as conv3d. Layout normalization to NDHWC is a
// separate decomposition pass, not the canonicalizer's job.
// NCDHW input (N=1, Cin=64, D=8, H=16, W=16), weight (Cout=128, Cin=64, 1,1,1).
func.func @negative_non_ndhwc(%input: tensor<1x64x8x16x16xbf16>,
                              %weight: tensor<128x64x1x1x1xbf16>) -> tensor<1x128x8x16x16xbf16> {
  // CHECK-LABEL: @negative_non_ndhwc
  // CHECK: "ttir.conv3d"
  // CHECK-NOT: "ttir.matmul"
  // CHECK-NOT: "ttir.linear"
  %0 = "ttir.conv3d"(%input, %weight)
          <{
            stride = array<i32: 1, 1, 1>,
            padding = array<i32: 0, 0, 0>,
            padding_mode = "zeros",
            groups = 1 : i32,
            batch_dim = 0 : i64,
            channel_dim = 1 : i64,
            depth_dim = 2 : i64,
            height_dim = 3 : i64,
            width_dim = 4 : i64
          }> : (tensor<1x64x8x16x16xbf16>, tensor<128x64x1x1x1xbf16>) -> tensor<1x128x8x16x16xbf16>
  return %0 : tensor<1x128x8x16x16xbf16>
}
