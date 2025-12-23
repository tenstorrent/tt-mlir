// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1 experimental-bfp8-weights=true enable-permute-matmul-fusion=false" -o gemma_2_2b_prefill_layer_ttnn.mlir %models/gemma_2_2b_prefill_layer.mlir
// RUN: FileCheck %s --input-file=gemma_2_2b_prefill_layer_ttnn.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn gemma_2_2b_prefill_layer_ttnn.mlir
// RUN: ttrt run --benchmark %t.ttnn
// CHECK-DAG: "ttnn.rms_norm"
