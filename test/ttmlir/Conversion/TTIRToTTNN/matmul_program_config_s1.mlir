// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --convert-ttir-to-ttnn -o %t %s
// RUN: FileCheck %s --input-file=%t

// Quetzal pilot: tt-metal's default matmul picker is bad for S=1 — it does NOT
// pick the 1D-multicast variant for M=1 big-matmul (LLM decode shape), leaving
// the DRAM-BW win on the table. MatmulOpConversionPattern emits a tuned
// MatmulMultiCoreReuseMultiCast1DProgramConfig for M=1 with K>=1024 and
// N>=1024, mirroring Quetzal's host-side codegen
// (ttnn_codegen.py:2050-2110). Small intermediate matmuls and the prefill
// (M>=32) shapes fall through to the default picker.

module {
  // M=1, K=4096, N=4096 — the canonical decode-time projection (q_proj /
  // o_proj on Llama-class shapes). Should carry the 1D mcast program_config.
  // CHECK-LABEL: func.func @matmul_decode_s1_emits_1d_program_config
  func.func @matmul_decode_s1_emits_1d_program_config(%arg0: tensor<1x4096xbf16>, %arg1: tensor<4096x4096xbf16>) -> tensor<1x4096xbf16> {
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config
    // CHECK-SAME: compute_with_storage_grid_size = #ttnn.core_coord<8, 8>
    // CHECK-SAME: per_core_m = 1
    // CHECK-SAME: mcast_in0 = true
    %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<1x4096xbf16>
    return %0 : tensor<1x4096xbf16>
  }

  // 3D batched M=1 — same decode shape with a leading batch dim. Effective M
  // is the penultimate dim of A, so this should still match.
  // CHECK-LABEL: func.func @matmul_decode_s1_3d_emits_1d_program_config
  func.func @matmul_decode_s1_3d_emits_1d_program_config(%arg0: tensor<1x1x2048xbf16>, %arg1: tensor<2048x8192xbf16>) -> tensor<1x1x8192xbf16> {
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config
    // CHECK-SAME: per_core_m = 1
    %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1x1x2048xbf16>, tensor<2048x8192xbf16>) -> tensor<1x1x8192xbf16>
    return %0 : tensor<1x1x8192xbf16>
  }

  // M=128 prefill shape — falls outside the decode trigger (M != 1), should
  // NOT carry a matmul_program_config, leaving the runtime picker in charge.
  // CHECK-LABEL: func.func @matmul_prefill_no_program_config
  func.func @matmul_prefill_no_program_config(%arg0: tensor<128x4096xbf16>, %arg1: tensor<4096x4096xbf16>) -> tensor<128x4096xbf16> {
    // CHECK: "ttnn.matmul"
    // CHECK-NOT: matmul_program_config
    %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<128x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<128x4096xbf16>
    return %0 : tensor<128x4096xbf16>
  }

  // Small N (below 1024) — even with M=1, the 1D-mcast layout doesn't pay off
  // and the heuristic must skip. Falls through to the default picker.
  // CHECK-LABEL: func.func @matmul_decode_small_n_no_program_config
  func.func @matmul_decode_small_n_no_program_config(%arg0: tensor<1x2048xbf16>, %arg1: tensor<2048x512xbf16>) -> tensor<1x512xbf16> {
    // CHECK: "ttnn.matmul"
    // CHECK-NOT: matmul_program_config
    %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1x2048xbf16>, tensor<2048x512xbf16>) -> tensor<1x512xbf16>
    return %0 : tensor<1x512xbf16>
  }

  // Small K (below 1024) — same gate on the contracting dim. The 1D-mcast
  // payoff comes from amortizing K-block dispatch across many cores; tiny K
  // is dominated by per-core overhead.
  // CHECK-LABEL: func.func @matmul_decode_small_k_no_program_config
  func.func @matmul_decode_small_k_no_program_config(%arg0: tensor<1x512xbf16>, %arg1: tensor<512x4096xbf16>) -> tensor<1x4096xbf16> {
    // CHECK: "ttnn.matmul"
    // CHECK-NOT: matmul_program_config
    %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1x512xbf16>, tensor<512x4096xbf16>) -> tensor<1x4096xbf16>
    return %0 : tensor<1x4096xbf16>
  }
}
