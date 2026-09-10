// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=2,4 optimization-level=1 experimental-bfp8-weights=true enable-permute-matmul-fusion=false enable-cpu-hoisted-const-eval=false" -o llama_3_1_70b_1layer_2x4mesh_ttnn.mlir %models/llama_3_1_70b_1layer_2x4mesh.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn llama_3_1_70b_1layer_2x4mesh_ttnn.mlir
// RUN: ttrt run %t.ttnn
