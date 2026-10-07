// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// The frontend forwards whatever numeric attribute the Python `scale` value
// produced, so an `int` scale arrives as an IntegerAttr and a `float` scale as
// a FloatAttr. Both must reach ttir.scaled_dot_product_attention: dropping one
// silently falls back to the 1/sqrt(head_dim) default, which is a wrong softmax
// temperature rather than an error.
module {
  // Integer scale (e.g. cosine attention's `self.scale = 1`).
  func.func @sdpa_int_scale(
      %q: tensor<1x16x5x128xbf16>,
      %k: tensor<1x16x5x128xbf16>,
      %v: tensor<1x16x5x128xbf16>,
      %mask: tensor<1x1x5x5xbf16>) -> tensor<1x16x5x128xbf16> {
    // CHECK-LABEL: func.func @sdpa_int_scale
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-SAME: scale = 1.000000e+00 : f32
    %0 = stablehlo.composite "tenstorrent.scaled_dot_product_attention" %q, %k, %v, %mask {
        composite_attributes = {is_causal = false, scale = 1 : i64},
        decomposition = @sdpa_impl
    } : (tensor<1x16x5x128xbf16>, tensor<1x16x5x128xbf16>, tensor<1x16x5x128xbf16>, tensor<1x1x5x5xbf16>) -> tensor<1x16x5x128xbf16>
    return %0 : tensor<1x16x5x128xbf16>
  }

  // Float scale.
  func.func @sdpa_float_scale(
      %q: tensor<1x16x5x128xbf16>,
      %k: tensor<1x16x5x128xbf16>,
      %v: tensor<1x16x5x128xbf16>,
      %mask: tensor<1x1x5x5xbf16>) -> tensor<1x16x5x128xbf16> {
    // CHECK-LABEL: func.func @sdpa_float_scale
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-SAME: scale = 0.0883883461 : f32
    %0 = stablehlo.composite "tenstorrent.scaled_dot_product_attention" %q, %k, %v, %mask {
        composite_attributes = {is_causal = false, scale = 0.0883883461 : f32},
        decomposition = @sdpa_impl
    } : (tensor<1x16x5x128xbf16>, tensor<1x16x5x128xbf16>, tensor<1x16x5x128xbf16>, tensor<1x1x5x5xbf16>) -> tensor<1x16x5x128xbf16>
    return %0 : tensor<1x16x5x128xbf16>
  }

  // No scale attribute -- the op keeps its 1/sqrt(head_dim) default.
  func.func @sdpa_no_scale(
      %q: tensor<1x16x5x128xbf16>,
      %k: tensor<1x16x5x128xbf16>,
      %v: tensor<1x16x5x128xbf16>,
      %mask: tensor<1x1x5x5xbf16>) -> tensor<1x16x5x128xbf16> {
    // CHECK-LABEL: func.func @sdpa_no_scale
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-NOT: scale =
    %0 = stablehlo.composite "tenstorrent.scaled_dot_product_attention" %q, %k, %v, %mask {
        composite_attributes = {is_causal = false},
        decomposition = @sdpa_impl
    } : (tensor<1x16x5x128xbf16>, tensor<1x16x5x128xbf16>, tensor<1x16x5x128xbf16>, tensor<1x1x5x5xbf16>) -> tensor<1x16x5x128xbf16>
    return %0 : tensor<1x16x5x128xbf16>
  }

  func.func private @sdpa_impl(
      %arg0: tensor<1x16x5x128xbf16>, %arg1: tensor<1x16x5x128xbf16>,
      %arg2: tensor<1x16x5x128xbf16>, %arg3: tensor<1x1x5x5xbf16>) -> tensor<1x16x5x128xbf16> {
    return %arg0 : tensor<1x16x5x128xbf16>
  }
}
