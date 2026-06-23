// RUN: rm -rf %t.dir && mkdir -p %t.dir
//
// Same model as llama_3_1_8b.mlir but lowered against a Blackhole
// system desc. Two variants (trace=off/on) must produce identical
// numbers, just like on Wormhole.
//
// TODO(#8767): the Blackhole TTIR→TTNN pipeline fails op-validation
// (worker-grid mismatch on ttnn.full) at optimization-level=1 for this
// IR. The failure is in the broader pipeline, not in our pass, so this
// test runs at the default opt level until the BH pipeline is fixed.
//
// trace=false
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mock-system-desc-arch=blackhole" \
// RUN:   --ttcore-unwrap-device-module \
// RUN:   --ttnn-collect-perf-metrics="ttnn-perf-metrics-output-file=%t.dir/no_trace.json" \
// RUN:   %models/single_blocks_and_layers/llama_3_1_8b_decode_layer.mlir -o /dev/null
// RUN: cat %t.dir/no_trace.json | FileCheck %s
//
// trace=true — same expected numbers
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mock-system-desc-arch=blackhole enable-trace=true" \
// RUN:   --ttcore-unwrap-device-module \
// RUN:   --ttnn-collect-perf-metrics="ttnn-perf-metrics-output-file=%t.dir/trace.json ttnn-enable-trace=true" \
// RUN:   %models/single_blocks_and_layers/llama_3_1_8b_decode_layer.mlir -o /dev/null
// RUN: cat %t.dir/trace.json | FileCheck %s

// Llama 3.1 8B single decoder layer on Blackhole. Different hardware
// constants than n150 (512 GB/s DRAM, 1.35 GHz, 130 Tensix cores) so the
// roofline shrinks vs the Wormhole companion test even though the model
// is the same. At default opt level SDPA is not fused, so 6 dram-bound
// weight matmuls + 2 skipped activation@activation matmuls. (The third,
// a degenerate K=1 dot_general, now lowers to a broadcast multiply rather
// than a matmul, so it is no longer counted among the skipped matmuls.)

// CHECK: "perf_targets":
// CHECK: "aiclk_hz": 1350000000
// CHECK: "arch": "blackhole"
// CHECK: "compute_bound_ops": 0
// CHECK: "dram_bandwidth_bytes_per_sec": 512000000000
// CHECK: "dram_bound_ops": 6
// CHECK: "num_chips": 1
// CHECK: "num_tensix_cores": 130
// CHECK: "params_count": 743440384
// CHECK: "params_memory_bytes": 789905408
// CHECK: "roofline_ms": 1.542784
// CHECK: "skipped_ops": 2
// CHECK: "top_perf_estimate_ms": 2.203977
