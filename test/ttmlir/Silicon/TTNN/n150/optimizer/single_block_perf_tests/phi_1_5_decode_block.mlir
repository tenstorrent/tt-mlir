// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1 experimental-bfp8-weights=true enable-permute-matmul-fusion=false" -o phi_1_5_decode_block_ttnn.mlir %models/phi_1_5_decode_block.mlir
// RUN: FileCheck %s --input-file=phi_1_5_decode_block_ttnn.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn phi_1_5_decode_block_ttnn.mlir
// RUN: ttrt run --benchmark %t.ttnn
// CHECK-DAG: "ttnn.layer_norm"
