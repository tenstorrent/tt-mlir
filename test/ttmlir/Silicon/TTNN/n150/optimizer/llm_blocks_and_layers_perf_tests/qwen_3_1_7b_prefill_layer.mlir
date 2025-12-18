// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1 experimental-bfp8-weights=true enable-permute-matmul-fusion=false" -o qwen_3_1_7b_prefill_layer_ttnn.mlir %models/llm_blocks_and_layers/qwen_3_1_7b_prefill_layer.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn qwen_3_1_7b_prefill_layer_ttnn.mlir
// RUN: ttrt run %t.ttnn
