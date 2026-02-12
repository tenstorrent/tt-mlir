// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=2 experimental-bfp8-weights=true enable-permute-matmul-fusion=false" -o falcon_3_1b_decode_layer_ttnn.mlir %models/single_blocks_and_layers/falcon_3_1b_decode_layer.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn falcon_3_1b_decode_layer_ttnn.mlir
// RUN: ttrt run --benchmark %t.ttnn
