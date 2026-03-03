// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=2 experimental-bfp8-weights=true enable-permute-matmul-fusion=false" -o mistral_7b_decode_layer_ttnn.mlir %models/single_blocks_and_layers/mistral_7b_decode_layer.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn mistral_7b_decode_layer_ttnn.mlir
// RUN: ttrt run --benchmark %t.ttnn
