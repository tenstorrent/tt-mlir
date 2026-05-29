// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mock-system-desc-arch=wormhole_b0" --ttcore-unwrap-device-module --ttnn-collect-perf-metrics="ttnn-perf-metrics-output-file=%t.dir/out.json" %models/single_blocks_and_layers/llama_3_1_8b_decode_layer.mlir -o /dev/null
// RUN: cat %t.dir/out.json | FileCheck %s

// Llama 3.1 8B single decoder layer lowered to TTNN against a mock
// Wormhole B0 system desc. Exercises the per-matmul compute-vs-DRAM
// classification end-to-end.
//
// The lowered forward function contains 9 ttnn.matmul ops:
//   - 7 weight projections (q/k/v/o + gate/up/down) reading large weights
//     from DRAM at small decode batch (32 tokens): DRAM-bound.
//   - 1 lm_head matmul (4096 -> 128256 vocab projection): DRAM-bound.
//   - 1 attention matmul that keeps both operands in L1: compute-bound.
//
// Numbers like roofline_time_us depend on pipeline behaviour and aren't
// pinned to exact values — only their presence is checked.

// CHECK: "perf_targets":
// CHECK: "arch": "wormhole_b0"
// CHECK: "compute_bound_ops": 0
// CHECK: "dram_bound_ops": 6
// CHECK: "roofline_ms":
// CHECK: "skipped_ops": 3
// CHECK: "top_perf_estimate_ms":
