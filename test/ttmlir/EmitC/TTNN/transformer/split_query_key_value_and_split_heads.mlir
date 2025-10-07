// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

func.func @split_heads_bge_m3(%arg0: tensor<2x34x3072xf32>) -> (tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>) {
  %0 = ttir.empty() : tensor<2x16x34x64xf32>
  %1 = ttir.empty() :  tensor<2x16x64x34xf32>
  %2 = ttir.empty() : tensor<2x16x34x64xf32>
  %3, %4, %5 = "ttir.split_query_key_value_and_split_heads"(%arg0, %0, %1, %2) <{num_heads = 16 : ui32, transpose_key = false}> : (tensor<2x34x3072xf32>, tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>) -> (tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>)
  return %3, %4, %5 : tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>
}
