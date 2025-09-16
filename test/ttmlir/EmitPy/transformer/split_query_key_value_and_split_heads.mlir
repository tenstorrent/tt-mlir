// RUN: ttmlir-opt --ttir-to-emitpy-pipeline -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir

func.func @split_qkv_and_split_heads(%arg0: tensor<2x32x3072xf32>) -> (tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>) {
  %0 = ttir.empty() : tensor<2x16x32x64xf32>
  %1 = ttir.empty() : tensor<2x16x32x64xf32>
  %2 = ttir.empty() : tensor<2x16x32x64xf32>
  %3, %4, %5 = "ttir.split_query_key_value_and_split_heads"(%arg0, %0, %1, %2) <{num_heads = 16 : ui32, transpose_key = false}> : (tensor<2x32x3072xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>) -> (tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>)
  return %3, %4, %5 : tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>
}
