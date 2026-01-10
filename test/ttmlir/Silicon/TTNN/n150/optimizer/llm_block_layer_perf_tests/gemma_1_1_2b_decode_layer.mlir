// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1 experimental-bfp8-weights=true enable-permute-matmul-fusion=false" -o gemma_1_1_2b_decode_layer_ttnn.mlir %models/llm_blocks_and_layers/gemma_1_1_2b_decode_layer.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn gemma_1_1_2b_decode_layer_ttnn.mlir
// RUN: ttrt run --benchmark %t.ttnn
