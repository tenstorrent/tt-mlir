// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1 experimental-bfp8-weights=true" -o llama_3_2_1b_1_layer_ttnn.mlir %models/llama_3_2_1b_1_layer.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn llama_3_2_1b_1_layer_ttnn.mlir
// RUN: ttrt run %t.ttnn
