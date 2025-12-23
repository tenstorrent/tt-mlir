// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1 experimental-bfp8-weights=true enable-permute-matmul-fusion=false" -o qwen_3_0_6b_prefill_layer_ttnn.mlir %models/qwen_3_0_6b_prefill_layer.mlir
// RUN: FileCheck %s --input-file=qwen_3_0_6b_prefill_layer_ttnn.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn qwen_3_0_6b_prefill_layer_ttnn.mlir
// RUN: ttrt run --benchmark %t.ttnn
// CHECK-DAG: "ttnn.rms_norm"
