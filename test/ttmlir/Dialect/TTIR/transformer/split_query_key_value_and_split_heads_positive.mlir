// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

module {
  func.func @split_qkv(%arg0: tensor<2x32x3072xf32>) -> (tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>) {
    %0 = ttir.empty() : tensor<2x16x32x64xf32>
    %1 = ttir.empty() : tensor<2x16x32x64xf32>
    %2 = ttir.empty() : tensor<2x16x32x64xf32>
    // CHECK: "ttnn.split_query_key_value_and_split_heads"(%arg0)
    %3, %4, %5  = "ttir.split_query_key_value_and_split_heads"(%arg0, %0, %1, %2) <{num_heads = 16 : ui32, transpose_key = false}> : (tensor<2x32x3072xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>) -> (tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>)
    return %3, %4, %5 : tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>
  }

  func.func @split_qkv_transpose(%arg0: tensor<2x32x3072xf32>) -> (tensor<2x16x32x64xf32>, tensor<2x16x64x32xf32>, tensor<2x16x32x64xf32>) {
    %0 = ttir.empty() : tensor<2x16x32x64xf32>
    %1 = ttir.empty() : tensor<2x16x64x32xf32>
    %2 = ttir.empty() : tensor<2x16x32x64xf32>
    // CHECK: "ttnn.split_query_key_value_and_split_heads"(%arg0)
    %3, %4, %5  = "ttir.split_query_key_value_and_split_heads"(%arg0, %0, %1, %2) <{num_heads = 16 : ui32, transpose_key = true}> : (tensor<2x32x3072xf32>, tensor<2x16x32x64xf32>, tensor<2x16x64x32xf32>, tensor<2x16x32x64xf32>) -> (tensor<2x16x32x64xf32>, tensor<2x16x64x32xf32>, tensor<2x16x32x64xf32>)
    return %3, %4, %5 : tensor<2x16x32x64xf32>, tensor<2x16x64x32xf32>, tensor<2x16x32x64xf32>
  }
}
