// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1 experimental-weight-dtype=bfp_bf8 enable-permute-matmul-fusion=false" -o llama_3_2_1b_1_layer_new_version_ttnn.mlir %models/llama_3_2_1b_1_layer_new_version.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn llama_3_2_1b_1_layer_new_version_ttnn.mlir
// RUN: ttrt run --benchmark %t.ttnn
