// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=2 experimental-weight-dtype=bfp_bf8 enable-permute-matmul-fusion=false" -o llama_3_1_8b_1lyr_bs32_decode_ttnn.mlir %models/single_blocks_and_layers/llama_3_1_8b_1lyr_bs32_decode.mlir
// RUN: FileCheck %s --input-file=llama_3_1_8b_1lyr_bs32_decode_ttnn.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn llama_3_1_8b_1lyr_bs32_decode_ttnn.mlir

// CHECK-DAG: "ttnn.rotary_embedding"
// CHECK-DAG: "ttnn.scaled_dot_product_attention_decode"
