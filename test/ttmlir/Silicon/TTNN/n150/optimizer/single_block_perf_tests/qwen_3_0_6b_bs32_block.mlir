// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1 experimental-bfp8-weights=true enable-permute-matmul-fusion=false" -o qwen_3_0_6b_bs32_block_ttnn.mlir %models/single_block/qwen_3_0_6b_bs32_block.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn qwen_3_0_6b_bs32_block_ttnn.mlir
// RUN: ttrt run %t.ttnn

