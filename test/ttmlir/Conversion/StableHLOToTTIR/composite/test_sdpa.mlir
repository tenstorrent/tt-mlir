// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // 3D mask must be left-padded to 4D before ttir.scaled_dot_product_attention.
  func.func @sdpa_3d_mask(
      %q: tensor<1x1x1280x1024xbf16>,
      %k: tensor<1x1x1280x1024xbf16>,
      %v: tensor<1x1x1280x1024xbf16>,
      %mask: tensor<1x1280x1280xbf16>) -> tensor<1x1x1280x1024xbf16> {
    // CHECK-LABEL: func.func @sdpa_3d_mask
    // CHECK: %[[MASK4D:[0-9]+]] = "ttir.reshape"(%arg3)
    // CHECK-SAME: shape = [1 : i32, 1 : i32, 1280 : i32, 1280 : i32]
    // CHECK-SAME: (tensor<1x1280x1280xbf16>{{.*}}) -> tensor<1x1x1280x1280xbf16>
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-SAME: %[[MASK4D]]
    %0 = stablehlo.composite "tenstorrent.scaled_dot_product_attention" %q, %k, %v, %mask {
        composite_attributes = {is_causal = false},
        decomposition = @sdpa_impl_3d
    } : (tensor<1x1x1280x1024xbf16>, tensor<1x1x1280x1024xbf16>, tensor<1x1x1280x1024xbf16>, tensor<1x1280x1280xbf16>) -> tensor<1x1x1280x1024xbf16>
    return %0 : tensor<1x1x1280x1024xbf16>
  }

  // 4D mask must pass through unchanged — no extra reshape inserted.
  func.func @sdpa_4d_mask(
      %q: tensor<1x1x1280x1024xbf16>,
      %k: tensor<1x1x1280x1024xbf16>,
      %v: tensor<1x1x1280x1024xbf16>,
      %mask: tensor<1x1x1280x1280xbf16>) -> tensor<1x1x1280x1024xbf16> {
    // CHECK-LABEL: func.func @sdpa_4d_mask
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttir.scaled_dot_product_attention"
    %0 = stablehlo.composite "tenstorrent.scaled_dot_product_attention" %q, %k, %v, %mask {
        composite_attributes = {is_causal = false},
        decomposition = @sdpa_impl_4d
    } : (tensor<1x1x1280x1024xbf16>, tensor<1x1x1280x1024xbf16>, tensor<1x1x1280x1024xbf16>, tensor<1x1x1280x1280xbf16>) -> tensor<1x1x1280x1024xbf16>
    return %0 : tensor<1x1x1280x1024xbf16>
  }

  func.func private @sdpa_impl_3d(
      %arg0: tensor<1x1x1280x1024xbf16>, %arg1: tensor<1x1x1280x1024xbf16>,
      %arg2: tensor<1x1x1280x1024xbf16>, %arg3: tensor<1x1280x1280xbf16>) -> tensor<1x1x1280x1024xbf16> {
    return %arg0 : tensor<1x1x1280x1024xbf16>
  }

  func.func private @sdpa_impl_4d(
      %arg0: tensor<1x1x1280x1024xbf16>, %arg1: tensor<1x1x1280x1024xbf16>,
      %arg2: tensor<1x1x1280x1024xbf16>, %arg3: tensor<1x1x1280x1280xbf16>) -> tensor<1x1x1280x1024xbf16> {
    return %arg0 : tensor<1x1x1280x1024xbf16>
  }
}
