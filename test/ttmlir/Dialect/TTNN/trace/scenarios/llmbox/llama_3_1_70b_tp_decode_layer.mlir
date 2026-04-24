// REQUIRES: opmodel, llmbox
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=2,4 optimization-level=1 enable-trace=true experimental-weight-dtype=bfp_bf8 enable-const-eval=true enable-cpu-hoisted-const-eval=true" -o llama_3_1_70b_tp_decode_layer_ttnn.mlir %models/single_blocks_and_layers/llama_3_1_70b_tp_decode_layer.mlir
// RUN: FileCheck %s --input-file=llama_3_1_70b_tp_decode_layer_ttnn.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn llama_3_1_70b_tp_decode_layer_ttnn.mlir

// Verify the expected fused / distributed ops are emitted by the pipeline.

// Input layernorm before the attention block.
// CHECK: "ttnn.distributed_rms_norm"

// Rotary embedding for Q and K projections.
// CHECK-COUNT-2: "ttnn.rotary_embedding"

// Decode-mode SDPA and the head-concat that follows it.
// CHECK: "ttnn.scaled_dot_product_attention_decode"
// CHECK: "ttnn.nlp_concat_heads_decode"

// Post-attention layernorm and the final model norm.
// CHECK-COUNT-2: "ttnn.distributed_rms_norm"

// Trace was enabled, so the program must be wrapped in a capture-or-execute trace call.
// CHECK: "ttnn.capture_or_execute_trace"
