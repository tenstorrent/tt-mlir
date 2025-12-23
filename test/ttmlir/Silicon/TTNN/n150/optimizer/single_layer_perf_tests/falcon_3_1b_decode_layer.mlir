// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1 experimental-bfp8-weights=true enable-permute-matmul-fusion=false" -o falcon_3_1b_decode_layer_ttnn.mlir %models/falcon_3_1b_decode_layer.mlir
// RUN: FileCheck %s --input-file=falcon_3_1b_decode_layer_ttnn.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn falcon_3_1b_decode_layer_ttnn.mlir
// RUN: ttrt run --benchmark %t.ttnn
// CHECK-DAG: "ttnn.layer_norm"

