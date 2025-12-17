// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1 experimental-bfp8-weights=true enable-permute-matmul-fusion=false" -o falcon_1b_bs32_block_ttnn.mlir %models/single_block/falcon_1b_bs32_block.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn falcon_1b_bs32_block_ttnn.mlir
// RUN: ttrt run %t.ttnn
