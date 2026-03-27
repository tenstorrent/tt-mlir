// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-greedy-optimizer=true experimental-bfp8-weights=true enable-permute-matmul-fusion=false" -o bert_encoder_layer_greedy_ttnn.mlir %models/single_blocks_and_layers/bert_encoder_layer.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn bert_encoder_layer_greedy_ttnn.mlir
// RUN: ttrt run --benchmark %t.ttnn
