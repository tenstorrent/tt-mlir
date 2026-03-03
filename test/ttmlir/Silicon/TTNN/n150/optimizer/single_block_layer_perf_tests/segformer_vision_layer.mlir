// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=2 experimental-bfp8-weights=true enable-permute-matmul-fusion=false" -o segformer_vision_layer_ttnn.mlir %models/single_blocks_and_layers/segformer_vision_layer.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn segformer_vision_layer_ttnn.mlir
// RUN: ttrt run --benchmark %t.ttnn
