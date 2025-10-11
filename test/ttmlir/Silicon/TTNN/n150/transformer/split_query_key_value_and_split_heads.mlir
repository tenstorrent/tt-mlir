// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module {
  func.func @forward_no_transpose(%input: tensor<8x16x3072xbf16>) -> (tensor<8x32x16x32xbf16>, tensor<8x32x16x32xbf16>, tensor<8x32x16x32xbf16>) {
    %0 = ttir.empty() : tensor<8x32x16x32xbf16>
    %1 = ttir.empty() : tensor<8x32x16x32xbf16>
    %2 = ttir.empty() : tensor<8x32x16x32xbf16>
    // CHECK: ttnn.split_query_key_value_and_split_heads
    %3, %4, %5 = "ttir.split_query_key_value_and_split_heads"(%input, %0, %1, %2) <{ num_heads = 32 : ui32, transpose_key = false }> : (tensor<8x16x3072xbf16>, tensor<8x32x16x32xbf16>, tensor<8x32x16x32xbf16>, tensor<8x32x16x32xbf16>) -> (tensor<8x32x16x32xbf16>, tensor<8x32x16x32xbf16>, tensor<8x32x16x32xbf16>)
    return %3, %4, %5 : tensor<8x32x16x32xbf16>, tensor<8x32x16x32xbf16>, tensor<8x32x16x32xbf16>
  }

  func.func @forward_transpose(%input: tensor<8x16x3072xbf16>) -> (tensor<8x32x16x32xbf16>, tensor<8x32x32x16xbf16>, tensor<8x32x16x32xbf16>) {
    %0 = ttir.empty() : tensor<8x32x16x32xbf16>
    %1 = ttir.empty() : tensor<8x32x32x16xbf16>
    %2 = ttir.empty() : tensor<8x32x16x32xbf16>
    // CHECK: ttnn.split_query_key_value_and_split_heads
    %3, %4, %5 = "ttir.split_query_key_value_and_split_heads"(%input, %0, %1, %2) <{ num_heads = 32 : ui32, transpose_key = true }> : (tensor<8x16x3072xbf16>, tensor<8x32x16x32xbf16>, tensor<8x32x32x16xbf16>, tensor<8x32x16x32xbf16>) -> (tensor<8x32x16x32xbf16>, tensor<8x32x32x16xbf16>, tensor<8x32x16x32xbf16>)
    return %3, %4, %5 : tensor<8x32x16x32xbf16>, tensor<8x32x32x16xbf16>, tensor<8x32x16x32xbf16>
  }

  func.func @forward_with_kv_input_tensor(%input : tensor<8x16x1024xbf16>, %input_kv : tensor<8x16x2048xbf16>) -> (tensor<8x32x16x32xbf16>, tensor<8x32x16x32xbf16>, tensor<8x32x16x32xbf16>) {
    %0 = ttir.empty() : tensor<8x32x16x32xbf16>
    %1 = ttir.empty() : tensor<8x32x16x32xbf16>
    %2 = ttir.empty() : tensor<8x32x16x32xbf16>
    // CHECK: ttnn.split_query_key_value_and_split_heads
    %3, %4, %5 = "ttir.split_query_key_value_and_split_heads"(%input, %input_kv, %0, %1, %2) <{ num_heads = 32 : ui32, transpose_key = false }> : (tensor<8x16x1024xbf16>, tensor<8x16x2048xbf16>, tensor<8x32x16x32xbf16>, tensor<8x32x16x32xbf16>, tensor<8x32x16x32xbf16>) -> (tensor<8x32x16x32xbf16>, tensor<8x32x16x32xbf16>, tensor<8x32x16x32xbf16>)
    return %3, %4, %5 : tensor<8x32x16x32xbf16>, tensor<8x32x16x32xbf16>, tensor<8x32x16x32xbf16>
  }
}
