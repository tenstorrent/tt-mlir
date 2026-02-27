// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=2 experimental-weight-dtype=bfp_bf8 enable-permute-matmul-fusion=false" -o llama_3_2_1b_decode_layer_ttnn.mlir %models/single_blocks_and_layers/llama_3_2_1b_decode_layer.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn llama_3_2_1b_decode_layer_ttnn.mlir
// RUN: ttrt run --benchmark %t.ttnn
