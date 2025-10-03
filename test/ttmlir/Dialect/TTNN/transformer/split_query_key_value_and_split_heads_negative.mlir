// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for split query key value and split heads operation.

// Verify that the parsing fails if input tensor is not 3D.
module {
  func.func @split_qkv_and_split_heads_invalid_1(%arg0: tensor<68x3072xf32>) -> (tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>) {
    %0 = ttir.empty() : tensor<2x16x34x64xf32>
    %1 = ttir.empty() :  tensor<2x16x64x34xf32>
    %2 = ttir.empty() : tensor<2x16x34x64xf32>
    // CHECK: error: 'ttir.split_query_key_value_and_split_heads' op expected rank of input tensor is 3, got rank 2
    %3, %4, %5 = "ttir.split_query_key_value_and_split_heads"(%arg0, %0, %1, %2) <{num_heads = 16 : ui32, transpose_key = false}> : (tensor<68x3072xf32>, tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>) -> (tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>)
    return %3, %4, %5 : tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>
  }
}
