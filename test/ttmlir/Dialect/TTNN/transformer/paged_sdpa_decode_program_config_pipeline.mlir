// Verify the workaround-injected SDPA program config survives the whole
// TTIR->TTNN backend pipeline (optimization-level=0 runs host-side).
//
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=0 mock-system-desc-arch=blackhole" -mlir-print-local-scope %s | FileCheck %s --check-prefix=BLACKHOLE
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=0 mock-system-desc-arch=wormhole_b0" -mlir-print-local-scope %s | FileCheck %s --check-prefix=WORMHOLE

module attributes {} {
  func.func @paged_sdpa_decode_causal(%arg0: tensor<1x1x12x64xf32>, %arg1: tensor<128x12x32x64xf32>, %arg2: tensor<128x12x32x64xf32>, %arg3: tensor<1x4xi32>, %arg4: tensor<1xi32>) -> tensor<1x1x12x64xf32> {
    // BLACKHOLE: ttnn.paged_scaled_dot_product_attention_decode
    // BLACKHOLE-SAME: q_chunk_size = 32
    // BLACKHOLE-SAME: k_chunk_size = 32
    // BLACKHOLE-SAME: exp_approx_mode = false
    // BLACKHOLE-SAME: max_cores_per_head_batch = 1
    //
    // On Wormhole a causal decode needs no program config.
    // WORMHOLE: ttnn.paged_scaled_dot_product_attention_decode
    // WORMHOLE-NOT: sdpa_program_config
    %0 = ttir.empty() : tensor<1x1x12x64xf32>
    %1 = "ttir.paged_scaled_dot_product_attention_decode"(%arg0, %arg1, %arg2, %arg3, %0, %arg4) <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1, 0>}> : (tensor<1x1x12x64xf32>, tensor<128x12x32x64xf32>, tensor<128x12x32x64xf32>, tensor<1x4xi32>, tensor<1x1x12x64xf32>, tensor<1xi32>) -> tensor<1x1x12x64xf32>
    return %1 : tensor<1x1x12x64xf32>
  }
}
