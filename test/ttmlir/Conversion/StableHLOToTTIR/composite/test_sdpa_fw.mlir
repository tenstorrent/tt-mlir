// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --legalize-stablehlo-composite-to-ttir -o %t %s
// RUN: FileCheck %s --input-file=%t

// SDPA forward composite with causal mask (3 operands, 1 result).
module {
  func.func @sdpa_fw(%query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
                     %value: tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16> {
    // CHECK: "ttir.sdpa_fw"
    // CHECK-SAME: mask_type = #ttcore.attention_mask_type<causal>
    // CHECK-SAME: return_intermediates = false
    // CHECK-NOT: stablehlo.composite
    %0 = stablehlo.composite "tenstorrent.sdpa_fw" %query, %key, %value {
      composite_attributes = {
        mask_type = 1 : i32,
        dropout_probability = 0.000000e+00 : f32
      },
      decomposition = @tenstorrent.sdpa_fw.impl
    } : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16>
    return %0 : tensor<1x8x64x64xbf16>
  }
  func.func private @tenstorrent.sdpa_fw.impl(%query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>, %value: tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16> {
    return %query : tensor<1x8x64x64xbf16>
  }
}

// -----

// SDPA forward composite with an arbitrary mask (4 operands) and intermediates.
module {
  func.func @sdpa_fw_arbitrary(%query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
                               %value: tensor<1x8x64x64xbf16>, %mask: tensor<1x1x64x64xbf16>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>) {
    // CHECK: "ttir.sdpa_fw"
    // CHECK-SAME: mask_type = #ttcore.attention_mask_type<arbitrary>
    // CHECK-SAME: return_intermediates = true
    // CHECK-NOT: stablehlo.composite
    %0:2 = stablehlo.composite "tenstorrent.sdpa_fw" %query, %key, %value, %mask {
      composite_attributes = {
        mask_type = 2 : i32,
        dropout_probability = 0.000000e+00 : f32
      },
      decomposition = @tenstorrent.sdpa_fw.arbitrary.impl
    } : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x1x64x64xbf16>) -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>)
    return %0#0, %0#1 : tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>
  }
  func.func private @tenstorrent.sdpa_fw.arbitrary.impl(%query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>, %value: tensor<1x8x64x64xbf16>, %mask: tensor<1x1x64x64xbf16>) -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>) {
    %0 = stablehlo.constant dense<0.0> : tensor<1x8x64x32xf32>
    return %query, %0 : tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>
  }
}
