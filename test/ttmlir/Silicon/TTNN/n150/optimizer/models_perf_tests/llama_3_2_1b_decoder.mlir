// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1 experimental-weight-dtype=bfp_bf8 enable-permute-matmul-fusion=false" -o llama_3_2_1b_decoder_ttnn.mlir %models/llama_3_2_1b_decoder.mlir
// RUN: FileCheck %s --input-file=llama_3_2_1b_decoder_ttnn.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn llama_3_2_1b_decoder_ttnn.mlir
// RUN: ttrt run --benchmark %t.ttnn
// CHECK-DAG: "ttnn.rms_norm"
