// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=2 experimental-weight-dtype=bfp_bf8 enable-permute-matmul-fusion=false" -o vit_vision_block_ttnn.mlir %models/single_blocks_and_layers/vit_vision_block.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn vit_vision_block_ttnn.mlir
// RUN: ttrt run --benchmark %t.ttnn
