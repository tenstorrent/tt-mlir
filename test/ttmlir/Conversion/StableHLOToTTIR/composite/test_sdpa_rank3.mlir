// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // 3D query/key/value must take a unit head dim at index 1, with the result
  // reshaped back to 3D.
  func.func @sdpa_rank3(
      %q: tensor<1x4096x1024xbf16>,
      %k: tensor<1x4096x1024xbf16>,
      %v: tensor<1x4096x1024xbf16>) -> tensor<1x4096x1024xbf16> {
    // CHECK-LABEL: func.func @sdpa_rank3(
    // CHECK: %[[Q4D:[0-9]+]] = "ttir.reshape"(%arg0)
    // CHECK-SAME: shape = [1 : i32, 1 : i32, 4096 : i32, 1024 : i32]
    // CHECK-SAME: (tensor<1x4096x1024xbf16>{{.*}}) -> tensor<1x1x4096x1024xbf16>
    // CHECK: %[[K4D:[0-9]+]] = "ttir.reshape"(%arg1)
    // CHECK-SAME: -> tensor<1x1x4096x1024xbf16>
    // CHECK: %[[V4D:[0-9]+]] = "ttir.reshape"(%arg2)
    // CHECK-SAME: -> tensor<1x1x4096x1024xbf16>
    // CHECK: %[[SDPA:[0-9]+]] = "ttir.scaled_dot_product_attention"(%[[Q4D]], %[[K4D]], %[[V4D]])
    // CHECK-SAME: -> tensor<1x1x4096x1024xbf16>
    // CHECK: "ttir.reshape"(%[[SDPA]])
    // CHECK-SAME: shape = [1 : i32, 4096 : i32, 1024 : i32]
    // CHECK-SAME: (tensor<1x1x4096x1024xbf16>{{.*}}) -> tensor<1x4096x1024xbf16>
    %0 = stablehlo.composite "tenstorrent.scaled_dot_product_attention" %q, %k, %v {
        composite_attributes = {is_causal = false},
        decomposition = @sdpa_impl_rank3
    } : (tensor<1x4096x1024xbf16>, tensor<1x4096x1024xbf16>, tensor<1x4096x1024xbf16>) -> tensor<1x4096x1024xbf16>
    return %0 : tensor<1x4096x1024xbf16>
  }

  // A 3D mask alongside 3D q/k/v takes the head dim at index 1, not a left pad.
  func.func @sdpa_rank3_mask(
      %q: tensor<2x4096x1024xbf16>,
      %k: tensor<2x4096x1024xbf16>,
      %v: tensor<2x4096x1024xbf16>,
      %mask: tensor<2x4096x4096xbf16>) -> tensor<2x4096x1024xbf16> {
    // CHECK-LABEL: func.func @sdpa_rank3_mask(
    // CHECK: %[[MASK4D:[0-9]+]] = "ttir.reshape"(%arg3)
    // CHECK-SAME: shape = [2 : i32, 1 : i32, 4096 : i32, 4096 : i32]
    // CHECK-SAME: (tensor<2x4096x4096xbf16>{{.*}}) -> tensor<2x1x4096x4096xbf16>
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-SAME: %[[MASK4D]]
    %0 = stablehlo.composite "tenstorrent.scaled_dot_product_attention" %q, %k, %v, %mask {
        composite_attributes = {is_causal = false},
        decomposition = @sdpa_impl_rank3_mask
    } : (tensor<2x4096x1024xbf16>, tensor<2x4096x1024xbf16>, tensor<2x4096x1024xbf16>, tensor<2x4096x4096xbf16>) -> tensor<2x4096x1024xbf16>
    return %0 : tensor<2x4096x1024xbf16>
  }

  // 4D query/key/value must pass through unchanged — no extra reshape inserted.
  func.func @sdpa_rank4(
      %q: tensor<1x1x4096x1024xbf16>,
      %k: tensor<1x1x4096x1024xbf16>,
      %v: tensor<1x1x4096x1024xbf16>) -> tensor<1x1x4096x1024xbf16> {
    // CHECK-LABEL: func.func @sdpa_rank4(
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttir.scaled_dot_product_attention"
    %0 = stablehlo.composite "tenstorrent.scaled_dot_product_attention" %q, %k, %v {
        composite_attributes = {is_causal = false},
        decomposition = @sdpa_impl_rank4
    } : (tensor<1x1x4096x1024xbf16>, tensor<1x1x4096x1024xbf16>, tensor<1x1x4096x1024xbf16>) -> tensor<1x1x4096x1024xbf16>
    return %0 : tensor<1x1x4096x1024xbf16>
  }

  func.func private @sdpa_impl_rank3(
      %arg0: tensor<1x4096x1024xbf16>, %arg1: tensor<1x4096x1024xbf16>,
      %arg2: tensor<1x4096x1024xbf16>) -> tensor<1x4096x1024xbf16> {
    return %arg0 : tensor<1x4096x1024xbf16>
  }

  func.func private @sdpa_impl_rank3_mask(
      %arg0: tensor<2x4096x1024xbf16>, %arg1: tensor<2x4096x1024xbf16>,
      %arg2: tensor<2x4096x1024xbf16>, %arg3: tensor<2x4096x4096xbf16>) -> tensor<2x4096x1024xbf16> {
    return %arg0 : tensor<2x4096x1024xbf16>
  }

  func.func private @sdpa_impl_rank4(
      %arg0: tensor<1x1x4096x1024xbf16>, %arg1: tensor<1x1x4096x1024xbf16>,
      %arg2: tensor<1x1x4096x1024xbf16>) -> tensor<1x1x4096x1024xbf16> {
    return %arg0 : tensor<1x1x4096x1024xbf16>
  }
}
