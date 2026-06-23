// optimization-level=1 enables the optimizer/fusing path, which requires
// OpModel support (otherwise the pipeline hits llvm_unreachable).
// REQUIRES: opmodel
// RUN: rm -rf %t.dir && mkdir -p %t.dir
//
// Two variants at optimization-level=1 — trace disabled and enabled. The
// pass walks the same set of weight-consuming ops in both, so all counts
// and the roofline must match exactly.
//
// trace=false
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mock-system-desc-arch=wormhole_b0 optimization-level=1" \
// RUN:   --ttcore-unwrap-device-module \
// RUN:   --ttnn-collect-perf-metrics="ttnn-perf-metrics-output-file=%t.dir/no_trace.json" \
// RUN:   %models/single_blocks_and_layers/llama_3_1_8b_decode_layer.mlir -o /dev/null
// RUN: cat %t.dir/no_trace.json | FileCheck %s
//
// trace=true — same expected numbers
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mock-system-desc-arch=wormhole_b0 optimization-level=1 enable-trace=true" \
// RUN:   --ttcore-unwrap-device-module \
// RUN:   --ttnn-collect-perf-metrics="ttnn-perf-metrics-output-file=%t.dir/trace.json ttnn-enable-trace=true" \
// RUN:   %models/single_blocks_and_layers/llama_3_1_8b_decode_layer.mlir -o /dev/null
// RUN: cat %t.dir/trace.json | FileCheck %s

// Llama 3.1 8B single decoder layer on Wormhole B0. SDPA is fused at opt=1
// so the 6 weight projections (q/k/v/o + gate/up/down) and the SDPA op
// itself are all DRAM-bound. The small ephemeral RoPE-helper, a degenerate
// K=1 dot_general, now lowers to a broadcast multiply rather than a matmul,
// so it is no longer counted as a skipped matmul (skipped_ops drops to 0).

// CHECK: "perf_targets":
// CHECK: "aiclk_hz": 1000000000
// CHECK: "arch": "wormhole_b0"
// CHECK: "compute_bound_ops": 0
// CHECK: "dram_bandwidth_bytes_per_sec": 288000000000
// CHECK: "dram_bound_ops": 7
// CHECK: "num_chips": 1
// CHECK: "num_tensix_cores": 64
// CHECK: "params_count": 751828992
// CHECK: "params_memory_bytes": 798818304
// CHECK: "roofline_ms": 2.773674
// CHECK: "skipped_ops": 0
// CHECK: "top_perf_estimate_ms": 3.962392
