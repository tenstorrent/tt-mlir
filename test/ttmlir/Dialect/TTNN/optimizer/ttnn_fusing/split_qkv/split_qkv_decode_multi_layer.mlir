// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mock-system-desc-arch=wormhole_b0 mesh-shape=1,2 optimization-level=1" --mlir-print-ir-after=ttnn-fusing %models/single_blocks_and_layers/llama_3_2_3b_tp_decode_two_layers.mlir 2>&1 | FileCheck %s

// Two-layer tensor-parallel decode slice (Llama-3.2-3B, n300), reduced from the
// reproducer in issue #8958.
//
// The decode QKV/RoPE fusion assigns Q/K/V roles by forward-tracing each chain
// to its SDPA operand. With more than one attention layer, that trace could run
// past a layer's SDPA into the next layer and reach a conflicting role, so role
// identification bailed and fell back to slice-position order. That fallback is
// wrong when the fused QKV weight is not already in Q,K,V order (here it is
// [V,K,Q]): it mislabels K and V, so RoPE lands on the *value* output and the
// key/value caches are swapped. The second layer here is the downstream SDPA
// that triggered the bleed for the first layer, so a single layer would not
// catch the regression.
//
// Guard: both layers must fuse, and RoPE (the rotate-half slices) must only
// touch query/key outputs, never a `value` output. Checked on the
// post-ttnn-fusing IR (before layout passes add memory-config ops).

// First (bleeding) layer: fuses, and RoPE must not touch its value output.
// CHECK: "ttnn.nlp_create_qkv_heads_decode"
// CHECK-NOT: "ttnn.slice_static"(%value
// CHECK-NOT: "ttnn.rotary_embedding"(%value
// Second (last) layer: also fuses, likewise no RoPE on its value output.
// CHECK: "ttnn.nlp_create_qkv_heads_decode"
// CHECK-NOT: "ttnn.slice_static"(%value
// CHECK-NOT: "ttnn.rotary_embedding"(%value
