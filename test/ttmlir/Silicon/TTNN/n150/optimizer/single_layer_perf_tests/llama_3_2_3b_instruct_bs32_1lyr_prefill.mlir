// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1 experimental-bfp8-weights=true enable-permute-matmul-fusion=false" -o llama_3_2_3b_instruct_bs32_1lyr_prefill_ttnn.mlir %models/single_layer/llama_3_2_3b_instruct_bs32_1lyr_prefill.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn llama_3_2_3b_instruct_bs32_1lyr_prefill_ttnn.mlir
// RUN: ttrt run --benchmark %t.ttnn
